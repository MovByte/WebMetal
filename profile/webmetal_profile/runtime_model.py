"""Handlers for the functions outside the executed images that the profiled code calls.

Each handler states the behaviour it models and its source: objc4 (`objc-runtime-new.mm`,
`NSObject.mm`, `objc-msg-arm64.s`) for the Objective-C runtime, libdispatch for
`dispatch_once`, the C standard for the memory and string functions. Behaviour that would
need state the profile does not have (I/O, the kernel, logging output) is not modelled.
"""

from .emu import EmulationError, ReportedFailure

ALL_ONES = (1 << 64) - 1


class ObjC:
    """Message dispatch and object lifetime over the cache's class metadata."""

    def __init__(self, machine):
        self.m = machine
        self.rt = machine.runtime
        self._selectors = None
        # (receiver, selector) -> handler(machine) returning the method's x0. Used for the
        # device methods whose result is state the kernel supplies at device creation.
        self.overrides = {}

    # selectors
    def selector(self, name):
        """The uniqued selector address the executed images use for name."""
        if self._selectors is None:
            table = {}
            cache = self.m.cache
            for _, _, image in self.m.executable:
                for _, start, size in image.section('__DATA', '__objc_selrefs'):
                    for at in range(start, start + size, 8):
                        address = cache.pointer(at)
                        table.setdefault(cache.cstring(address), address)
            self._selectors = table
        if name not in self._selectors:
            raise EmulationError(f'no uniqued selector {name} referenced by the executed images')
        return self._selectors[name]

    def class_of(self, obj):
        return self.rt.klass(self.m.u64(obj))

    def defined_by_root(self, klass, selector):
        cls, method = self.rt.lookup(klass, selector)
        if method is None:
            return True
        image = self.m.cache.image_at(method.imp)
        return image is not None and image.name == 'libobjc.A.dylib'

    def has_custom_awz(self, cls_address):
        meta = self.rt.klass(self.m.u64(cls_address))
        return not (self.defined_by_root(meta, 'alloc') and self.defined_by_root(meta, 'allocWithZone:'))

    def ensure_initialized(self, cls_address):
        """+initialize, superclass first, as lookUpImpOrForward/initializeNonMetaClass do."""
        klass = self.rt.klass(cls_address)
        if klass.is_meta:
            raise EmulationError('ensure_initialized on a metaclass')
        for c in reversed(self.rt.chain(klass)):
            if c.address in self.m.initialized_classes:
                continue
            self.m.initialized_classes.add(c.address)
            meta = self.rt.klass(c.isa)
            _, method = self.rt.lookup(meta, 'initialize')
            if method is None:
                continue
            image = self.m.cache.image_at(method.imp)
            if image is not None and image.name == 'libobjc.A.dylib':
                continue  # +[NSObject initialize] does nothing
            self.m.call(method.imp, (c.address, self.selector('initialize')))

    def receiver_class(self, receiver):
        klass = self.class_of(receiver)
        if klass.is_meta:
            self.ensure_initialized(receiver)
        else:
            self.ensure_initialized(klass.address)
        return klass

    def dispatch(self, receiver, selector, start=None):
        """IMP address (or handler result) for [receiver selector]."""
        klass = self.receiver_class(receiver) if start is None else start
        cls, method = self.rt.lookup(klass, selector)
        if method is None:
            raise EmulationError(f'-[{klass.name} {selector}]: no method (forwarding is not modelled)')
        self.m.calls.append((selector, receiver, cls.name))
        return method.imp

    def send(self, receiver, selector, *args):
        """Nested [receiver selector args...] from a handler; returns x0."""
        if receiver == 0:
            return 0
        imp = self.dispatch(receiver, selector)
        return self.m.call(imp, (receiver, self.selector(selector)) + args)

    def allocate(self, cls_address):
        klass = self.rt.klass(cls_address)
        if klass.is_meta:
            raise EmulationError('allocation of a metaclass instance')
        return self.m.new_object(klass)


def install(m):
    objc = ObjC(m)
    m.objc = objc
    h = m.handlers
    env_log = m.environment_queries = []

    def imp_or_handler(imp):
        """Jump to an executable IMP; run the handler of a runtime-implemented IMP."""
        if m._code_owner(imp) == 'allowed':
            return imp
        image, name, offset = m.cache.symbolize(imp)
        handler = h.get(name)
        if handler is None or offset:
            raise EmulationError(f'method implementation {image.name if image else "?"}`{name} is not modelled')
        return handler(m)

    # objc_msgSend (objc-msg-arm64.s): nil receiver returns zero in x0, x1 and v0-v3.
    def msg_send(m):
        receiver = m.x(0)
        if receiver == 0:
            m.set_x(1, 0)
            for i in range(4):
                m.set_q(i, 0)
            return m.return_value(0)
        selector = m.cstring(m.x(1))
        override = objc.overrides.get((receiver, selector))
        if override is not None:
            m.calls.append((selector, receiver, 'override'))
            return m.return_value(override(m))
        return imp_or_handler(objc.dispatch(receiver, selector))
    h['_objc_msgSend'] = msg_send

    # objc_msgSendSuper2(struct objc_super *super, SEL op, ...): lookup starts at
    # super->super_class->superclass; the receiver is super->receiver.
    def msg_send_super2(m):
        sup = m.x(0)
        receiver, current = m.u64(sup), m.u64(sup + 8)
        m.set_x(0, receiver)
        klass = objc.rt.klass(current)
        selector = m.cstring(m.x(1))
        start = objc.rt.klass(klass.superclass)
        return imp_or_handler(objc.dispatch(receiver, selector, start))
    h['_objc_msgSendSuper2'] = msg_send_super2

    # Reference counting has no observable effect on the values profiled: identity.
    for name in ('_objc_retain', '_objc_autorelease', '_objc_retainAutorelease',
                 '_objc_autoreleaseReturnValue', '_objc_retainAutoreleaseReturnValue',
                 '_objc_retainAutoreleasedReturnValue', '_objc_claimAutoreleasedReturnValue',
                 '_objc_unsafeClaimAutoreleasedReturnValue', '_objc_retainBlock'):
        h[name] = lambda m: None
    for reg in range(1, 29):
        h[f'_objc_retain_x{reg}'] = (lambda r: lambda m: m.return_value(m.x(r)))(reg)
        h[f'_objc_release_x{reg}'] = lambda m: None
    h['_objc_release'] = lambda m: None
    h['-[NSObject retain]'] = lambda m: None
    h['-[NSObject release]'] = lambda m: None
    h['-[NSObject autorelease]'] = lambda m: None

    def store_strong(m):
        m.put64(m.x(0), m.x(1))
        return None
    h['_objc_storeStrong'] = store_strong

    # Allocation (NSObject.mm callAlloc / _objc_rootAllocWithZone / class_createInstance).
    def objc_alloc(m):
        cls = m.x(0)
        if cls == 0:
            return m.return_value(0)
        objc.ensure_initialized(cls)
        if objc.has_custom_awz(cls):
            return m.return_value(objc.send(cls, 'alloc'))
        return m.return_value(objc.allocate(cls))
    h['_objc_alloc'] = objc_alloc

    def objc_alloc_with_zone(m):
        cls = m.x(0)
        if cls == 0:
            return m.return_value(0)
        objc.ensure_initialized(cls)
        meta = objc.rt.klass(m.u64(cls))
        if not objc.defined_by_root(meta, 'allocWithZone:'):
            return m.return_value(objc.send(cls, 'allocWithZone:', 0))
        return m.return_value(objc.allocate(cls))
    h['_objc_allocWithZone'] = objc_alloc_with_zone

    def objc_alloc_init(m):
        cls = m.x(0)
        if cls == 0:
            return m.return_value(0)
        objc.ensure_initialized(cls)
        obj = objc.send(cls, 'alloc') if objc.has_custom_awz(cls) else objc.allocate(cls)
        return m.return_value(objc.send(obj, 'init'))
    h['_objc_alloc_init'] = objc_alloc_init

    def objc_opt_new(m):
        cls = m.x(0)
        if cls == 0:
            return m.return_value(0)
        objc.ensure_initialized(cls)
        meta = objc.rt.klass(m.u64(cls))
        if not objc.defined_by_root(meta, 'new'):
            return m.return_value(objc.send(cls, 'new'))
        obj = objc.send(cls, 'alloc') if objc.has_custom_awz(cls) else objc.allocate(cls)
        return m.return_value(objc.send(obj, 'init'))
    h['_objc_opt_new'] = objc_opt_new

    def nsobject_alloc(m):
        cls = m.x(0)
        meta = objc.rt.klass(m.u64(cls))
        if not objc.defined_by_root(meta, 'allocWithZone:'):
            return m.return_value(objc.send(cls, 'allocWithZone:', 0))
        return m.return_value(objc.allocate(cls))
    h['+[NSObject alloc]'] = nsobject_alloc
    h['+[NSObject allocWithZone:]'] = lambda m: m.return_value(objc.allocate(m.x(0)))
    h['-[NSObject init]'] = lambda m: None
    h['+[NSObject class]'] = lambda m: None
    h['-[NSObject class]'] = lambda m: m.return_value(objc.class_of(m.x(0)).address)

    def opt_class(m):
        obj = m.x(0)
        if obj == 0:
            return m.return_value(0)
        klass = objc.class_of(obj)
        return m.return_value(obj if klass.is_meta else klass.address)
    h['_objc_opt_class'] = opt_class

    def opt_is_kind(m):
        obj, cls = m.x(0), m.x(1)
        if obj == 0:
            return m.return_value(0)
        return m.return_value(int(any(c.address == cls for c in objc.rt.chain(objc.class_of(obj)))))
    h['_objc_opt_isKindOfClass'] = opt_is_kind

    def opt_responds(m):
        obj, sel = m.x(0), m.x(1)
        if obj == 0:
            return m.return_value(0)
        _, method = objc.rt.lookup(objc.class_of(obj), m.cstring(sel))
        return m.return_value(int(method is not None))
    h['_objc_opt_respondsToSelector'] = opt_responds

    # libdispatch dispatch_once: runs the block once, then marks the predicate DLOCK_ONCE_DONE (~0).
    def dispatch_once(m):
        predicate, block = m.x(0), m.x(1)
        if m.u64(predicate) != ALL_ONES:
            m.call(m.u64(block + 16), (block,))
            m.put64(predicate, ALL_ONES)
        return None
    h['_dispatch_once'] = dispatch_once

    def dispatch_once_f(m):
        predicate, context, function = m.x(0), m.x(1), m.x(2)
        if m.u64(predicate) != ALL_ONES:
            m.call(function, (context,))
            m.put64(predicate, ALL_ONES)
        return None
    h['_dispatch_once_f'] = dispatch_once_f

    # C allocator (memory content of malloc is unspecified; the model zero-fills).
    h['_malloc'] = lambda m: m.return_value(m.malloc(m.x(0)))
    h['_calloc'] = lambda m: m.return_value(m.malloc(m.x(0) * m.x(1)))
    h['_malloc_type_malloc'] = lambda m: m.return_value(m.malloc(m.x(0)))
    h['_malloc_type_calloc'] = lambda m: m.return_value(m.malloc(m.x(0) * m.x(1)))
    h['__Znwm'] = lambda m: m.return_value(m.malloc(m.x(0)))
    h['__Znam'] = lambda m: m.return_value(m.malloc(m.x(0)))
    h['__ZnwmSt19__type_descriptor_t'] = lambda m: m.return_value(m.malloc(m.x(0)))
    h['__ZnamSt19__type_descriptor_t'] = lambda m: m.return_value(m.malloc(m.x(0)))
    for name in ('_free', '__ZdlPv', '__ZdaPv', '__ZdlPvm'):
        h[name] = lambda m: None

    def realloc(m):
        old, size = m.x(0), m.x(1)
        new = m.malloc(size)
        if old:
            m.write(new, m.read(old, min(size, m.allocations.get(old, 0))))
        return m.return_value(new)
    h['_realloc'] = realloc
    h['_malloc_type_realloc'] = realloc

    def memset(m):
        m.write(m.x(0), bytes([m.x(1) & 0xff]) * m.x(2))
        return None
    for name in ('_memset', '__platform_memset'):
        h[name] = memset

    def bzero(m):
        m.write(m.x(0), bytes(m.x(1)))
        return None
    for name in ('_bzero', '__platform_bzero'):
        h[name] = bzero

    def memmove(m):
        if m.x(2):
            m.write(m.x(0), m.read(m.x(1), m.x(2)))
        return None
    for name in ('_memcpy', '_memmove', '__platform_memmove'):
        h[name] = memmove

    h['_strlen'] = lambda m: m.return_value(len(m.cstring(m.x(0), 1 << 20).encode()))

    def strcmp(m):
        a, b = m.cstring(m.x(0)).encode(), m.cstring(m.x(1)).encode()
        return m.return_value((a > b) - (a < b))
    h['_strcmp'] = strcmp

    # Environment: the profile describes a process with no Metal environment overrides.
    def getenv(m):
        env_log.append(m.cstring(m.x(0)))
        return m.return_value(0)
    h['_getenv'] = getenv

    # dyld_program_sdk_at_least(dyld_build_version_t): the profile describes a program built
    # with the iOS 17 SDK. dyld_build_version_t is {uint32_t platform; uint32_t version} and
    # arrives packed in x0 (platform in the low half). platform 0xffffffff compares version
    # sets, whose version word is (year << 16 | month << 8 | day): sets of a year before 2023
    # are older than any iOS 17 SDK, sets after 2024 are newer than any. platform 2 (iOS)
    # compares X.Y.Z as 0xXXXXYYZZ: below 17.0 is older, above 17.7 newer. Anything between is
    # ambiguous for "an iOS 17 SDK" and stops the run.
    sdk_log = m.sdk_queries = []

    def program_sdk_at_least(m):
        raw = m.x(0)
        platform, version = raw & 0xffffffff, raw >> 32
        caller = m.describe(m.lr - 4)
        if platform == 0xffffffff:
            year = version >> 16
            if year < 2023:
                answer = True
            elif year > 2024:
                answer = False
            else:
                raise EmulationError(f'dyld_program_sdk_at_least version set {version:#x} from {caller} '
                                     'is ambiguous for an iOS 17 SDK')
        elif platform == 2:
            if version < 0x110000:
                answer = True
            elif version > 0x110700:
                answer = False
            else:
                raise EmulationError(f'dyld_program_sdk_at_least iOS {version:#x} from {caller} '
                                     'is ambiguous for an iOS 17 SDK')
        else:
            raise EmulationError(f'dyld_program_sdk_at_least platform {platform} from {caller}')
        sdk_log.append((platform, version, answer, caller))
        return m.return_value(int(answer))
    h['_dyld_program_sdk_at_least'] = program_sdk_at_least

    # os_log: logging disabled for every type; nothing observable depends on log output.
    h['_os_log_type_enabled'] = lambda m: m.return_value(0)

    # Failure paths of the profiled code are reported, never continued. MTLReportFailure
    # (type, function, line, format, ...) and MTLReleaseAssertionFailure (function, line,
    # condition, ...) carry the source line that identifies the failed check.
    def report_failure(m):
        raise ReportedFailure('MTLReportFailure', m.x(2) & 0xffffffff, m.describe(m.lr - 4))
    h['_MTLReportFailure'] = report_failure

    def release_assertion(m):
        raise ReportedFailure('MTLReleaseAssertionFailure', m.x(1) & 0xffffffff, m.describe(m.lr - 4))
    h['_MTLReleaseAssertionFailure'] = release_assertion

    for name in ('_abort', '___assert_rtn', '___cxa_throw', '_objc_exception_throw', '___stack_chk_fail'):
        h[name] = (lambda n: lambda m: (_ for _ in ()).throw(
            EmulationError(f'{n} reached from {m.describe(m.lr - 4)}')))(name)
