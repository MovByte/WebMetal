// Depth-stencil states: WebMetalDepthStencilState stands in for AGXA10FamilyDepthStencilState
// (AGXA10FamilyDepthStencilState -> _MTLDepthStencilState on 21H461).
//
// The driver packs the descriptor into one 64-bit word, derives from it which of the depth and stencil values a draw
// reads and writes (the four answers below), and encodes the A10X's depth-stencil words, which its GPU reads. The
// plugin's GPU is the host's, which builds its own form of the state from the descriptor (CreateDepthStencilState).
// The driver keeps every state it makes in a device-wide cache keyed by the packed word and the label, so equal
// descriptors give the same object; the plugin keeps the same cache (WebMetalDepthStencilCache.h).

#import "WebMetalResources.h"
#import "WebMetalDepthStencilCache.h"
#import "../Commands/WebMetalCommands.h"

// The class adopts MTLDepthStencilState and MTLDepthStencilStateSPI as AGXA10FamilyDepthStencilState does and
// implements only what the 21H461 driver implements; the protocols' other methods are Metal.framework's or LOUD stubs.
#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wobjc-protocol-property-synthesis"

@interface _MTLDepthStencilState (WebMetalFramework)
- (instancetype)initWithDevice:(id<MTLDevice>)device depthStencilDescriptor:(MTLDepthStencilDescriptor *)descriptor;
@end

@interface WebMetalDepthStencilState : _MTLDepthStencilState <MTLDepthStencilStateSPI>
@end

@interface WebMetalDepthStencilState () {
@public
    uint64_t _handle;
    // AGXA10FamilyDepthStencilState._impl: the packed descriptor (+0x0) and the load/store needs (+0x20).
    uint64_t _packed;
    uint32_t _loadStoreNeeds;
}
@end

namespace webmetal {
namespace {

struct Packed {
    uint64_t key;
    uint32_t loadStoreNeeds;
};

struct Face {
    uint64_t compare, fail, depthFail, pass;
    uint32_t readMask, writeMask;
};

Face face(MTLStencilDescriptor *s)
{
    return Face{s.stencilCompareFunction, s.stencilFailureOperation, s.depthFailureOperation,
                s.depthStencilPassOperation, s.readMask, s.writeMask};
}

// The packing and derivation of -[AGXA10FamilyDevice newDepthStencilStateWithDescriptor:] (0x21c8d5410-0x21c8d5768).
// The low word holds the front face (compare bits 0-2, stencil-fail, depth-fail and pass operations in bits 3-11), the
// back face (bits 13-24 alike), the depth compare function (bits 26-28) and depth writes (bit 29); bytes 4-7 are the
// low bytes of the front read and write masks and the back read and write masks. A face's stencil test is enabled
// (bit 12, bit 25) unless it always passes and never writes. A state whose stencil operations all keep and whose depth
// compare function is MTLCompareFunctionNotEqual is replaced by the driver's canonical form of it.
//
// The needs word: bit 0 reads depth (a depth compare function other than never or always, or depth writes), bit 1
// writes depth, bit 2 reads stencil, bit 3 writes stencil; bits 16 and 17 are the driver's depth-test flags.
Packed pack(const Face &f, const Face &b, uint64_t depthCompare, bool depthWrite)
{
    const uint32_t lo = uint32_t(f.compare & 7) | uint32_t(f.fail & 7) << 3 | uint32_t(f.depthFail & 7) << 6 |
                        uint32_t(f.pass & 7) << 9 | uint32_t(b.compare & 7) << 13 | uint32_t(b.fail & 7) << 16 |
                        uint32_t(b.depthFail & 7) << 19 | uint32_t(b.pass & 7) << 22 |
                        uint32_t(depthCompare & 7) << 26 | (depthWrite ? 1u << 29 : 0u);
    const uint64_t masks = uint64_t(f.readMask & 0xff) | uint64_t(f.writeMask & 0xff) << 8 |
                           uint64_t(b.readMask & 0xff) << 16 | uint64_t(b.writeMask & 0xff) << 24;
    const bool frontOps = lo & 0xff8, backOps = lo & 0x1ff0000;
    const uint32_t frontCompare = lo & 7, backCompare = lo >> 13 & 7;
    const bool frontWriteMask = f.writeMask & 0xff, backWriteMask = b.writeMask & 0xff;
    const bool frontEnabled = frontCompare != 7 || (frontWriteMask && frontOps);
    const bool backEnabled = backCompare != 7 || (backWriteMask && backOps);
    const bool frontWrites = frontOps && frontWriteMask, backWrites = backOps && backWriteMask;
    bool frontReads = frontWrites, backReads = backWrites;
    if (!frontWrites && frontEnabled)
        frontReads = (f.readMask & 0xff) && frontCompare != 7 && frontCompare != 0;
    if (!backWrites && backEnabled)
        backReads = (b.readMask & 0xff) && backCompare != 7 && backCompare != 0;
    const uint32_t depth = lo >> 26 & 7, write = lo >> 29 & 1;
    const uint32_t low = (depth != 7 && depth != 0 ? 1 : 0) | write | write << 1 |
                         (frontWrites || backWrites || frontReads || backReads ? 4 : 0) |
                         (frontWrites || backWrites ? 8 : 0);
    uint32_t word = (lo & 0xfdffefff) | (frontEnabled ? 0x1000 : 0) | (backEnabled ? 0x2000000 : 0);
    uint64_t key = word | masks << 32;
    if ((lo & 0x1dff0ff8) == 0x14000000) {
        word = (lo & 0xfdf8efc7) | 0x2021010;
        key = (word | masks << 32) & 0x00ff00ffffffffffull;
    }
    bool stencilKeeps;
    if (!(word & 0x1000) || !(word & 0x1f8))
        stencilKeeps = !(word & 0x2000000) || (!(word & 0x70000) && !(word & 0x380000));
    else
        stencilKeeps = false;
    uint32_t high = write ^ 1;
    if ((word & 0x1c000000) == 0x1c000000)
        high = 1;
    if ((word & 0x1c000000) == 0x14000000 && stencilKeeps)
        high |= 2;
    return Packed{key, low | high << 16};
}

wire::StencilDescriptor wireFace(MTLStencilDescriptor *s)
{
    wire::StencilDescriptor w;
    w.depthFailureOperation = s.depthFailureOperation;
    w.depthStencilPassOperation = s.depthStencilPassOperation;
    w.readMask = s.readMask;
    w.stencilCompareFunction = s.stencilCompareFunction;
    w.stencilFailureOperation = s.stencilFailureOperation;
    w.writeMask = s.writeMask;
    return w;
}

}  // namespace

// -[AGXA10FamilyDevice newDepthStencilStateWithDescriptor:] (AGXMetalA10 0x21c8d53e0): under the cache's lock, the
// descriptor is packed and looked up with its label (an empty string for none); a hit returns the cached state, a miss
// makes one with -[_MTLDepthStencilState initWithDevice:depthStencilDescriptor:], stores the packed word and the needs
// in it, and adds it to the cache, which keeps its reference. The returned state carries a reference for the caller
// either way. Nothing is refused. A state holds its device (_MTLDepthStencilState) and the device's cache holds the
// state, so a state lives as long as its device and its handle stays registered.
id<MTLDepthStencilState> NewDepthStencilState(WebMetalDevice *device, SEL _cmd, MTLDepthStencilDescriptor *d)
{
    DepthStencilCache &cache = DeviceDepthStencilStates(device);
    os_unfair_lock_lock(&cache.lock);
    MTLStencilDescriptor *front = d.frontFaceStencil, *back = d.backFaceStencil;
    const Packed packed = pack(face(front), face(back), d.depthCompareFunction, d.isDepthWriteEnabled);
    NSString *label = d.label;
    std::pair<uint64_t, std::string> key{packed.key, label ? std::string(label.UTF8String) : std::string()};
    auto found = cache.states.find(key);
    if (found != cache.states.end()) {
        id state = found->second.state;
        os_unfair_lock_unlock(&cache.lock);
        return state;
    }
    WebMetalDepthStencilState *state = [[WebMetalDepthStencilState alloc] initWithDevice:device
                                                                   depthStencilDescriptor:d];
    state->_packed = packed.key;
    state->_loadStoreNeeds = packed.loadStoreNeeds;
    Connection *connection = DeviceConnection(device);
    state->_handle = connection->NewHandle();
    wire::CreateDepthStencilState request;
    request.state = state->_handle;
    wire::DepthStencilDescriptor &w = request.descriptor;
    if (front) {
        w.frontFaceStencil.present = true;
        w.frontFaceStencil.value = wireFace(front);
    }
    if (back) {
        w.backFaceStencil.present = true;
        w.backFaceStencil.value = wireFace(back);
    }
    w.depthCompareFunction = d.depthCompareFunction;
    w.depthWriteEnabled = d.isDepthWriteEnabled;
    if (!wire::strFromNSString(label, w.label))
        WMLoudMethod(device, _cmd, "the label has no UTF-8 form, which the wire format carries");
    wire::Reply<wire::CreateDepthStencilStateReply> reply;
    connection->Call(device, _cmd, request, &reply);
    if (!reply.ok)
        WMLoudMethod(device, _cmd, "the host answered with a Metal error, which the A10X driver does not return here");
    RegisterHandle(state, state->_handle);
    cache.states.emplace(std::move(key), DepthStencilCache::Entry{packed.loadStoreNeeds, state});
    os_unfair_lock_unlock(&cache.lock);
    return state;
}

}  // namespace webmetal

@implementation WebMetalDepthStencilState

// -[AGXA10FamilyDepthStencilState readsDepth] (AGXMetalA10 0x21c8d1a3c) and the three below: bits 0-3 of the needs.
- (BOOL)readsDepth
{
    return _loadStoreNeeds & 1;
}

- (BOOL)writesDepth
{
    return _loadStoreNeeds >> 1 & 1;
}

- (BOOL)readsStencil
{
    return _loadStoreNeeds >> 2 & 1;
}

- (BOOL)writesStencil
{
    return _loadStoreNeeds >> 3 & 1;
}

@end
