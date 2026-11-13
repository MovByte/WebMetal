// The command queue's GPU priorities (AGXA10FamilyCommandQueue and IOGPUMetalCommandQueue on 21H461).
//
// IOGPU keeps the priority and the background priority the kernel last applied (_priority and _backgroundPriority)
// and changes them through IOGPUCommandQueueSetPriorityAndBackground (IOGPU 0x1fcb2ed18): IOGPUDeviceUserClient
// selector 27 with the queue's ID and the 32-bit priority and background priority, whose scalar output is the
// priority the kernel applied. The 21H461 kernel's handler (0xfffffff005bbbf88) finds the queue and calls its
// setter (0xfffffff005bd7d44), which accepts priorities 0, 1, 2 and 4 as given, applies 5 (MTLGPUPriorityRealTime)
// only for a task entitled com.apple.private.agx.rt-priority-spi and 0 otherwise (logging that it does), refuses any
// other priority and every background priority but 2 and 3 with kIOReturnBadArgument, and changes the queue's
// scheduling. The plugin's queues are host queues over one WebGPU queue, where a priority changes nothing; the kernel's
// answer depends only on the arguments and the task's entitlements, so the plugin computes it here.

#import "WebMetalSPI.h"
#import "../Commands/WebMetalCommandState.h"

// The class adopts the driver's protocols; these categories implement the driver selectors they declare.
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"

namespace {

// IOGPU's validateGPUPriority (0x1fcb2eba8): a priority above 5 is no MTLGPUPriority, and 3 is a background priority.
void validateGPUPriority(NSUInteger priority, bool background)
{
    if (priority > 5)
        MTLReportFailure(0, "validateGPUPriority", 122, @"priority (%d) is not a valid MTLGPUPriority", (int)priority);
    else if (priority == 3 && !background)
        MTLReportFailure(0, "validateGPUPriority", 119, @"priority (%d) is not valid for foreground", 3);
}

}  // namespace

@implementation WebMetalCommandQueue (WebMetalSPI)

// -[AGXA10FamilyCommandQueue resolveAndUpdateQueuePriority:] (AGXMetalA10 0x21c8ca240): on a customer build the
// priority as given, which the driver also keeps as its current priority (read by nothing the plugin reproduces). On
// an internal build the driver first applies AGX_FORCE_GPU_PRIORITY.
- (NSUInteger)resolveAndUpdateQueuePriority:(NSUInteger)priority
{
    if (webmetal::InternalBuild(self, _cmd))
        WMLoudMethod(self, _cmd, "on an internal build the A10X driver forces the priority from AGX_FORCE_GPU_PRIORITY "
                                 "(findEnvVarNum, 0x21c9d7044), which the plugin does not reproduce");
    return priority;
}

// -[IOGPUMetalCommandQueue _setGPUPriority:backgroundPriority:] (IOGPU 0x1fcb2eca0): the kernel's answer, kept on
// success.
- (BOOL)_setGPUPriority:(NSUInteger)priority backgroundPriority:(NSUInteger)background
{
    const uint32_t requested = static_cast<uint32_t>(priority);
    uint32_t applied;
    if (requested < 3 || requested == 4)
        applied = requested;
    else if (requested == 5)
        applied = webmetal::TaskHasEntitlement("com.apple.private.agx.rt-priority-spi") ? 5 : 0;
    else
        return NO;
    if ((static_cast<uint32_t>(background) & ~1u) != 2)
        return NO;
    _gpuPriority = applied;
    _backgroundGPUPriority = background;
    return YES;
}

// -[AGXA10FamilyCommandQueue setGPUPriority:] (AGXMetalA10 0x21c8ca3c0) resolves the priority and calls
// -[IOGPUMetalCommandQueue setGPUPriority:] (IOGPU 0x1fcb2eb68): validate it for the foreground, then set it with
// the current background priority.
- (BOOL)setGPUPriority:(NSUInteger)priority
{
    const NSUInteger resolved = [self resolveAndUpdateQueuePriority:priority];
    validateGPUPriority(resolved, false);
    return [self _setGPUPriority:resolved backgroundPriority:_backgroundGPUPriority];
}

// -[AGXA10FamilyCommandQueue setGPUPriority:offset:] (AGXMetalA10 0x21c8ca408) and -[IOGPUMetalCommandQueue
// setGPUPriority:offset:] (IOGPU 0x1fcb3335c): as -setGPUPriority:; IOGPU passes no offset to the kernel.
- (BOOL)setGPUPriority:(NSUInteger)priority offset:(uint16_t)offset
{
    const NSUInteger resolved = [self resolveAndUpdateQueuePriority:priority];
    validateGPUPriority(resolved, false);
    return [self _setGPUPriority:resolved backgroundPriority:_backgroundGPUPriority];
}

// -[IOGPUMetalCommandQueue setBackgroundGPUPriority:] (IOGPU 0x1fcb30b0c): report a priority above 5, then set the
// background priority with the current priority.
- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority
{
    if (priority > 5)
        MTLReportFailure(0, "validateGPUPriority", 122, @"priority (%d) is not a valid MTLGPUPriority", (int)priority);
    return [self _setGPUPriority:_gpuPriority backgroundPriority:priority];
}

// -[IOGPUMetalCommandQueue setBackgroundGPUPriority:offset:] (IOGPU 0x1fcb3339c): as -setBackgroundGPUPriority:.
- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority offset:(uint16_t)offset
{
    if (priority > 5)
        MTLReportFailure(0, "validateGPUPriority", 122, @"priority (%d) is not a valid MTLGPUPriority", (int)priority);
    return [self _setGPUPriority:_gpuPriority backgroundPriority:priority];
}

// -[IOGPUMetalCommandQueue getGPUPriority] (IOGPU 0x1fcb30c60).
- (NSUInteger)getGPUPriority
{
    return _gpuPriority;
}

// -[IOGPUMetalCommandQueue getBackgroundGPUPriority] (IOGPU 0x1fcb30c70).
- (NSUInteger)getBackgroundGPUPriority
{
    return _backgroundGPUPriority;
}

// The A10 family's queues have no statistics counters: -[AGXA10FamilyCommandQueue availableCounters], -counterInfo,
// -getRequestedCounters and -subdivideCounterList: (AGXMetalA10 0x21c8ca6e8, 0x21c8ca700, 0x21c8ca6f8, 0x21c8ca708)
// answer nil, -requestCounters: (0x21c8ca6f0) answers -1 and -setStatEnabled: (0x21c8ca710) returns at once.
- (NSArray *)availableCounters
{
    return nil;
}

- (NSDictionary *)counterInfo
{
    return nil;
}

- (NSArray *)getRequestedCounters
{
    return nil;
}

- (NSArray *)subdivideCounterList:(NSArray *)counters
{
    return nil;
}

- (int)requestCounters:(NSArray *)counters
{
    return -1;
}

- (void)setStatEnabled:(BOOL)enabled
{
}

@end
