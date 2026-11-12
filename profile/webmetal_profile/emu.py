"""Execute code from a dyld shared cache under Unicorn with an explicit Objective-C model.

Only code in the images named at construction runs as machine code (plus the stub islands
that sit between images). A branch into any other image stops emulation; the target is
symbolized and served by a handler registered for that symbol, or the run fails with
`EmulationError` naming the symbol. Handlers model the Objective-C runtime (message dispatch
through the cache's own class metadata, allocation, reference counting as identity),
`dispatch_once`, the C allocator and the C string/memory primitives. Nothing else is
modelled: an unlisted call is an error, never a default return value.

Objects created with `new_object(track=True)` record which bytes the emulated code has
written. A read of a byte that no emulated code wrote raises `UninitializedRead` with the
instance variable it belongs to, so a result can never silently depend on state that the
real initializer would have set.
"""

import struct

from unicorn import (UC_ARCH_ARM64, UC_HOOK_CODE, UC_HOOK_MEM_READ, UC_HOOK_MEM_UNMAPPED,
                     UC_HOOK_MEM_WRITE, UC_MODE_ARM, UC_PROT_EXEC, UC_PROT_READ, UC_PROT_WRITE,
                     Uc, UcError)
from unicorn import arm64_const as A

PAGE = 0x1000
HEAP_BASE, HEAP_SIZE = 0x300000000, 0x10000000
STACK_BASE, STACK_SIZE = 0x2f0000000, 0x200000
RETURN_BASE = 0x2e0000000
STEP_LIMIT = 50_000_000


class EmulationError(Exception):
    pass


class UninitializedRead(EmulationError):
    def __init__(self, obj, offset, size, pc, where):
        self.obj, self.offset, self.size, self.pc, self.where = obj, offset, size, pc, where
        super().__init__(f'read of unwritten bytes {offset:#x}+{size} of {obj.label} '
                         f'({where}) at pc {pc:#x}')


class ReportedFailure(EmulationError):
    """The profiled code reached Metal's own failure reporting (a validation failure)."""

    def __init__(self, function, line, caller):
        self.function, self.line, self.caller = function, line, caller
        super().__init__(f'{function} (line {line}) from {caller}')


X = [getattr(A, f'UC_ARM64_REG_X{i}') for i in range(29)]
D = [getattr(A, f'UC_ARM64_REG_D{i}') for i in range(32)]
Q = [getattr(A, f'UC_ARM64_REG_Q{i}') for i in range(32)]


class TrackedObject:
    def __init__(self, machine, address, size, klass, label):
        self.address, self.size, self.klass, self.label = address, size, klass, label
        self.written = bytearray(size)
        self.reads = []
        self.machine = machine

    def where(self, offset):
        rt = self.machine.runtime
        if self.klass is None:
            return 'untyped'
        if offset < 8:
            return 'isa'
        cls, ivar = rt.ivar_at(self.klass, offset)
        if ivar is None:
            return 'padding'
        return f'{cls.name}.{ivar.name}+{offset - ivar.offset:#x}'


class Machine:
    def __init__(self, cache, runtime, executable_images, trace=False):
        self.cache, self.runtime = cache, runtime
        self.trace = trace
        self.uc = uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
        # FP/SIMD enabled at EL0 (CPACR_EL1.FPEN = 0b11).
        uc.reg_write(A.UC_ARM64_REG_CPACR_EL1, uc.reg_read(A.UC_ARM64_REG_CPACR_EL1) | (3 << 20))
        uc.mem_map(HEAP_BASE, HEAP_SIZE, UC_PROT_READ | UC_PROT_WRITE)
        uc.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_READ | UC_PROT_WRITE)
        uc.mem_map(RETURN_BASE, PAGE, UC_PROT_READ)
        self.heap_top = HEAP_BASE
        self.executable = sorted((image.address, image.address + image.text_size, image)
                                 for image in executable_images)
        self.executable_paths = {image.path for image in executable_images}
        self.mapped = set()
        self.handlers = {}
        self.depth = 0
        self.tracked = []
        self._pending = None
        self._fault = None
        self.allocations = {}
        self.initialized_classes = set()
        self.selector_addresses = {}
        self.calls = []            # (selector or symbol, receiver, result) log for callers
        uc.hook_add(UC_HOOK_MEM_UNMAPPED, self._on_unmapped)
        from . import runtime_model
        runtime_model.install(self)

    # -- memory -----------------------------------------------------------------------------
    def _code_owner(self, address):
        """'allowed' for executable images and islands, else the owning Image."""
        image = self.cache.image_at(address)
        if image is None or image.path in self.executable_paths:
            return 'allowed'
        return image

    def _map_page(self, page):
        if page in self.mapped:
            return True
        m = self.cache.mapping(page)
        if m is None or page + PAGE > m.end:
            return False
        prot = 0
        if m.init_prot & 1:
            prot |= UC_PROT_READ
        if m.init_prot & 2:
            prot |= UC_PROT_WRITE
        stop_ranges = []
        if m.init_prot & 4:
            # Execute only what belongs to an executable image or to no image (stub islands).
            owners = {}
            for at in range(page, page + PAGE, 4):
                owners.setdefault(self._code_owner(at), []).append(at)
            if set(owners) == {'allowed'}:
                prot |= UC_PROT_EXEC
            elif 'allowed' in owners:
                prot |= UC_PROT_EXEC
                for owner, addresses in owners.items():
                    if owner != 'allowed':
                        stop_ranges.append((addresses[0], addresses[-1] + 3))
        self.uc.mem_map(page, PAGE, prot)
        self.uc.mem_write(page, self.cache.materialize(page, PAGE))
        for begin, end in stop_ranges:
            self.uc.hook_add(UC_HOOK_CODE, self._on_foreign_code, begin=begin, end=end)
        self.mapped.add(page)
        return True

    def intercept(self, address):
        """Stop at address even inside an executable image, so its handler runs instead."""
        self.uc.hook_add(UC_HOOK_CODE, self._on_foreign_code, begin=address, end=address)

    def _on_foreign_code(self, uc, address, size, _):
        self._fault = address
        uc.emu_stop()

    def _on_unmapped(self, uc, access, address, size, value, _):
        first, last = address & ~(PAGE - 1), (address + max(size, 1) - 1) & ~(PAGE - 1)
        for page in range(first, last + PAGE, PAGE):
            if not self._map_page(page):
                return False
        return True

    def read(self, address, size):
        self.ensure(address, size)
        return bytes(self.uc.mem_read(address, size))

    def write(self, address, data):
        self.ensure(address, len(data))
        self.uc.mem_write(address, bytes(data))

    def ensure(self, address, size):
        first, last = address & ~(PAGE - 1), (address + max(size, 1) - 1) & ~(PAGE - 1)
        for page in range(first, last + PAGE, PAGE):
            if not (HEAP_BASE <= page < HEAP_BASE + HEAP_SIZE or STACK_BASE <= page < STACK_BASE + STACK_SIZE
                    or page == RETURN_BASE):
                if not self._map_page(page):
                    raise EmulationError(f'access to unbacked address {address:#x}')

    def u64(self, address):
        return struct.unpack('<Q', self.read(address, 8))[0]

    def u32(self, address):
        return struct.unpack('<I', self.read(address, 4))[0]

    def put64(self, address, value):
        self.write(address, struct.pack('<Q', value & (1 << 64) - 1))

    def cstring(self, address, limit=4096):
        out = bytearray()
        while len(out) < limit:
            chunk = self.read(address + len(out), 1)
            if chunk == b'\0':
                return out.decode('utf-8')
            out += chunk
        raise EmulationError(f'unterminated string at {address:#x}')

    # -- heap -------------------------------------------------------------------------------
    def malloc(self, size, zero=True, align=16):
        size = max(size, 1)
        address = (self.heap_top + align - 1) // align * align
        if address + size > HEAP_BASE + HEAP_SIZE:
            raise EmulationError('emulated heap exhausted')
        self.heap_top = address + size
        self.allocations[address] = size
        if zero:
            self.uc.mem_write(address, bytes(size))
        return address

    def new_object(self, klass, track=False, label=None):
        """An instance of klass as objc's class_createInstance makes it: zeroed, isa set."""
        address = self.malloc(klass.instance_size)
        self.put64(address, klass.address)
        if track:
            obj = TrackedObject(self, address, klass.instance_size, klass, label or klass.name)
            obj.written[0:8] = b'\1' * 8
            self.tracked.append(obj)
            self.uc.hook_add(UC_HOOK_MEM_WRITE, self._on_tracked_write, obj, begin=address,
                             end=address + klass.instance_size - 1)
            self.uc.hook_add(UC_HOOK_MEM_READ, self._on_tracked_read, obj, begin=address,
                             end=address + klass.instance_size - 1)
            return obj
        return address

    def _on_tracked_write(self, uc, access, address, size, value, obj):
        start = max(address, obj.address) - obj.address
        end = min(address + size, obj.address + obj.size) - obj.address
        obj.written[start:end] = b'\1' * (end - start)

    def _on_tracked_read(self, uc, access, address, size, value, obj):
        start = max(address, obj.address) - obj.address
        end = min(address + size, obj.address + obj.size) - obj.address
        if not all(obj.written[start:end]):
            first = start + obj.written[start:end].find(0)
            pc = uc.reg_read(A.UC_ARM64_REG_PC)
            if self._pending is None:
                self._pending = UninitializedRead(obj, first, end - first, pc, obj.where(first))
            uc.emu_stop()

    # -- registers --------------------------------------------------------------------------
    def x(self, i):
        return self.uc.reg_read(X[i])

    def set_x(self, i, value):
        self.uc.reg_write(X[i], value & (1 << 64) - 1)

    @property
    def lr(self):
        return self.uc.reg_read(A.UC_ARM64_REG_LR)

    @property
    def pc(self):
        return self.uc.reg_read(A.UC_ARM64_REG_PC)

    def d(self, i):
        return self.uc.reg_read(D[i])

    def set_d(self, i, value):
        self.uc.reg_write(D[i], value)

    def set_q(self, i, value):
        self.uc.reg_write(Q[i], value)

    # -- execution --------------------------------------------------------------------------
    def call(self, function, args=(), x8=None, sp=None):
        """Call function(args...) and return x0. Nested calls are allowed from handlers."""
        uc = self.uc
        saved = None
        if self.depth:
            saved = uc.context_save()
        self.depth += 1
        if self.depth > 64:
            raise EmulationError('emulated call depth bound')
        ret = RETURN_BASE + 4 * self.depth
        try:
            if sp is None:
                sp = (uc.reg_read(A.UC_ARM64_REG_SP) - 0x400) & ~0xf if saved else STACK_BASE + STACK_SIZE - 0x100
            uc.reg_write(A.UC_ARM64_REG_SP, sp)
            for i, value in enumerate(args):
                self.set_x(i, value)
            if x8 is not None:
                self.set_x(8, x8)
            uc.reg_write(A.UC_ARM64_REG_LR, ret)
            pc = function
            steps = 0
            while True:
                if pc == ret:
                    break
                self._pending = None
                self._fault = None
                fetch_fault = False
                try:
                    uc.emu_start(pc, RETURN_BASE + PAGE, count=STEP_LIMIT)
                except UcError as error:
                    if self._pending is None and 'FETCH' not in str(error):
                        stopped = uc.reg_read(A.UC_ARM64_REG_PC)
                        raise EmulationError(f'{error} at pc {stopped:#x} ({self.describe(stopped)})')
                    fetch_fault = True
                if self._pending is not None:
                    raise self._pending
                if self._fault is not None:
                    stopped = self._fault
                elif fetch_fault:
                    stopped = uc.reg_read(A.UC_ARM64_REG_PC)
                else:
                    stopped = uc.reg_read(A.UC_ARM64_REG_PC)
                    raise EmulationError(f'step limit reached at {stopped:#x} ({self.describe(stopped)})')
                if stopped == ret:
                    break
                if RETURN_BASE <= stopped < RETURN_BASE + PAGE:
                    raise EmulationError(f'return to a stale frame {stopped:#x}')
                pc = self._handle(stopped)
            return self.x(0)
        finally:
            self.depth -= 1
            if saved is not None:
                result = (self.x(0), self.x(1), [self.uc.reg_read(Q[i]) for i in range(4)])
                uc.context_restore(saved)
                self._last = result
            else:
                self._last = (self.x(0), self.x(1), [self.uc.reg_read(Q[i]) for i in range(4)])

    def describe(self, address):
        image, name, offset = self.cache.symbolize(address)
        if image is None:
            return 'no image'
        return f'{image.name}`{name}+{offset:#x}'

    def _handle(self, address):
        image, name, offset = self.cache.symbolize(address)
        if image is None:
            raise EmulationError(f'execution stopped at {address:#x} outside every image')
        if offset != 0:
            raise EmulationError(f'branch into the middle of {image.name}`{name}+{offset:#x}')
        handler = self.handlers.get(name)
        if handler is None:
            raise EmulationError(f'call to unmodelled function {image.name}`{name} '
                                 f'from {self.describe(self.lr - 4)}')
        target = handler(self)
        return self.lr if target is None else target

    def return_value(self, value):
        self.set_x(0, value)
        return None
