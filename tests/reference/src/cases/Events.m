// Event, fence, hazard-tracking, debug-label, counter and IO-queue cases.

#import "WMRefCase.h"

static BOOL EventsWait(dispatch_semaphore_t semaphore, double seconds) {
    return dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(seconds * NSEC_PER_SEC))) == 0;
}

static void EventsSignalOnCompletion(id<MTLCommandBuffer> commandBuffer, dispatch_semaphore_t semaphore) {
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        dispatch_semaphore_signal(semaphore);
    }];
}

static void EventsFill(id<MTLCommandBuffer> commandBuffer, id<MTLBuffer> buffer, uint8_t value) {
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:value];
    [blit endEncoding];
}

static void EventsCopy(id<MTLCommandBuffer> commandBuffer, id<MTLBuffer> source, id<MTLBuffer> destination) {
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:source.length];
    [blit endEncoding];
}

#pragma mark - Events

// MTLCommandBuffer.h, encodeWaitForEvent:value: "pauses execution of this
// command buffer until the specified event reaches a given value";
// encodeSignalEvent:value: "signals an event with a given value".
WMREF_CASE(crossQueueEvent, "event.cross-queue-signal-wait", "event.ordering",
           "A command buffer waiting on an MTLEvent runs after a buffer on another queue signals it")
{
    id<MTLEvent> event = [c.device newEvent];
    if (!event) [c fail:@"newEvent returned nil"];
    [c value:@"event.device-is-creator" boolean:event.device == c.device rule:WMExact()];
    id<MTLCommandQueue> waitingQueue = [c.device newCommandQueue], signallingQueue = [c.device newCommandQueue];
    id<MTLBuffer> source = [c buffer:1024], destination = [c buffer:1024];

    id<MTLCommandBuffer> waiting = [waitingQueue commandBuffer];
    [waiting encodeWaitForEvent:event value:1];
    EventsCopy(waiting, source, destination);
    dispatch_semaphore_t waitingDone = dispatch_semaphore_create(0);
    EventsSignalOnCompletion(waiting, waitingDone);
    [waiting commit];
    [c step:@"waiting-committed"];

    // The waiting buffer cannot complete before the event is signalled.
    BOOL completedEarly = EventsWait(waitingDone, 0.3);
    [c value:@"waiting.completed-before-signal-commit" boolean:completedEarly rule:WMExact()];
    // A committed buffer is committed or scheduled until it completes
    // (MTLCommandBuffer.h); which of the two is not specified.
    MTLCommandBufferStatus whileWaiting = waiting.status;
    [c value:@"waiting.committed-or-scheduled-while-waiting"
        boolean:whileWaiting == MTLCommandBufferStatusCommitted || whileWaiting == MTLCommandBufferStatusScheduled
        rule:WMExact()];

    id<MTLCommandBuffer> signalling = [signallingQueue commandBuffer];
    EventsFill(signalling, source, 0x5a);
    [signalling encodeSignalEvent:event value:1];
    dispatch_semaphore_t signallingDone = dispatch_semaphore_create(0);
    EventsSignalOnCompletion(signalling, signallingDone);
    [signalling commit];
    if (!completedEarly && !EventsWait(waitingDone, 10)) [c fail:@"waiting buffer did not complete within 10 s"];
    if (!EventsWait(signallingDone, 10)) [c fail:@"signalling buffer did not complete within 10 s"];
    [c value:@"waiting.status" u64:waiting.status rule:WMExact()];
    [c value:@"signalling.status" u64:signalling.status rule:WMExact()];
    [c data:@"destination" elem:WMElemU8 bytes:destination.contents length:destination.length rule:WMExact()];
}

// MTLEvent.h, MTLSharedEvent: signaledValue "Read or set signaled value";
// waitUntilSignaledValue:timeoutMS: "Returns YES if the value was signaled
// before the timeout, otherwise NO."
WMREF_CASE(sharedEventCPUSignal, "event.shared-cpu-signal", "event.shared",
           "MTLSharedEvent signalled from the CPU releases a GPU wait; GPU signal sets signaledValue")
{
    id<MTLSharedEvent> event = [c.device newSharedEvent];
    if (!event) [c fail:@"newSharedEvent returned nil"];
    [c value:@"initial-signaled-value" u64:event.signaledValue rule:WMExact()];
    id<MTLBuffer> target = [c buffer:512];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    [commandBuffer encodeWaitForEvent:event value:2];
    EventsFill(commandBuffer, target, 0x3c);
    [commandBuffer encodeSignalEvent:event value:5];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    EventsSignalOnCompletion(commandBuffer, done);
    [commandBuffer commit];

    event.signaledValue = 1;
    [c value:@"after-cpu-signal-1.signaled-value" u64:event.signaledValue rule:WMExact()];
    BOOL earlyAt1 = EventsWait(done, 0.3);
    [c value:@"completed-with-value-1" boolean:earlyAt1 rule:WMExact()];
    event.signaledValue = 2;
    if (!earlyAt1 && !EventsWait(done, 10)) [c fail:@"command buffer did not complete within 10 s"];
    [c value:@"status" u64:commandBuffer.status rule:WMExact()];
    [c value:@"after-gpu-signal.signaled-value" u64:event.signaledValue rule:WMExact()];
    [c data:@"target" elem:WMElemU8 bytes:target.contents length:target.length rule:WMExact()];
    [c value:@"wait-until-5" boolean:[event waitUntilSignaledValue:5 timeoutMS:1000] rule:WMExact()];
    [c value:@"wait-until-100-timeout-20ms" boolean:[event waitUntilSignaledValue:100 timeoutMS:20] rule:WMExact()];
}

// MTLEvent.h, notifyListener:atValue:block: "When the event's signaled
// value reaches value or higher, invoke the block on the dispatch queue
// owned by the listener."
WMREF_CASE(sharedEventListener, "event.shared-listener", "event.shared",
           "notifyListener:atValue:block: for values reached before, at and after registration")
{
    id<MTLSharedEvent> event = [c.device newSharedEvent];
    if (!event) [c fail:@"newSharedEvent returned nil"];
    dispatch_queue_t queue = dispatch_queue_create("listener", DISPATCH_QUEUE_SERIAL);
    MTLSharedEventListener *listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:queue];
    NSMutableArray<NSString *> *log = [NSMutableArray array];
    dispatch_semaphore_t fired = dispatch_semaphore_create(0);
    MTLSharedEventNotificationBlock (^note)(NSString *) = ^MTLSharedEventNotificationBlock(NSString *name) {
        return ^(id<MTLSharedEvent> e, uint64_t value) {
            @synchronized(log) {
                [log addObject:[NSString stringWithFormat:@"%@:%llu:%llu", name, value, e.signaledValue]];
            }
            dispatch_semaphore_signal(fired);
        };
    };
    event.signaledValue = 3;
    // Registered after the event already reached the value.
    [event notifyListener:listener atValue:2 block:note(@"at-2")];
    if (!EventsWait(fired, 5)) [c fail:@"a listener for a value already reached did not run within 5 s"];
    @synchronized(log) {
        [c value:@"already-reached" strings:[log copy] rule:WMExact()];
        [log removeAllObjects];
    }
    [event notifyListener:listener atValue:4 block:note(@"at-4")];
    [event notifyListener:listener atValue:6 block:note(@"at-6")];
    [event notifyListener:listener atValue:9 block:note(@"at-9")];
    BOOL earlyFire = EventsWait(fired, 0.2);
    [c value:@"fired-before-value-reached" boolean:earlyFire rule:WMExact()];
    // A GPU signal to 6 reaches the first two values.
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    [commandBuffer encodeSignalEvent:event value:6];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    for (int i = 0; i < 2; i++)
        if (!EventsWait(fired, 5)) [c fail:@"listeners for 4 and 6 did not run within 5 s"];
    BOOL extra = EventsWait(fired, 0.2);
    [c value:@"fired-beyond-signaled-value" boolean:extra rule:WMExact()];
    @synchronized(log) {
        // Entries are "name:notified value:signaledValue at the time"; the
        // value the block receives is the registered value or later.
        NSArray *sorted = [log sortedArrayUsingSelector:@selector(compare:)];
        [c value:@"after-gpu-signal-6.sorted" strings:sorted
            rule:WMNone(@"the value passed and signaledValue observed by a block are not documented")];
        NSMutableArray *names = [NSMutableArray array];
        for (NSString *entry in sorted) [names addObject:[entry componentsSeparatedByString:@":"][0]];
        [c value:@"after-gpu-signal-6.notified" strings:names rule:WMExact()];
        [log removeAllObjects];
    }
    event.signaledValue = 10;
    if (!EventsWait(fired, 5)) [c fail:@"listener for 9 did not run within 5 s"];
    @synchronized(log) {
        NSMutableArray *names = [NSMutableArray array];
        for (NSString *entry in log) [names addObject:[entry componentsSeparatedByString:@":"][0]];
        [c value:@"after-cpu-signal-10.notified" strings:names rule:WMExact()];
    }
}

// MTLEvent.h: "MTLSharedEventHandle objects may be passed between processes
// via XPC connections and then used to recreate a MTLSharedEvent via an
// existing MTLDevice." The handle encodes only through an XPC coder; a keyed
// archiver is refused by the framework.
WMREF_CASE(sharedEventHandle, "event.shared-handle", "event.shared",
           "newSharedEventHandle and newSharedEventWithHandle: share one value; keyed archiving is refused")
{
    id<MTLSharedEvent> event = [c.device newSharedEvent];
    if (!event) [c fail:@"newSharedEvent returned nil"];
    event.label = @"shared-event";
    MTLSharedEventHandle *handle = [event newSharedEventHandle];
    if (!handle) [c fail:@"newSharedEventHandle returned nil"];
    [c value:@"handle.label" string:handle.label rule:WMExact()];
    NSError *error = nil;
    NSData *archive = [NSKeyedArchiver archivedDataWithRootObject:handle requiringSecureCoding:YES error:&error];
    [c value:@"keyed-archiver.refused" boolean:archive == nil rule:WMExact()];
    [c value:@"keyed-archiver.error-domain" string:error.domain rule:WMExact()];
    [c value:@"keyed-archiver.error-code" i64:error.code rule:WMExact()];
    id<MTLSharedEvent> first = [c.device newSharedEventWithHandle:handle];
    id<MTLSharedEvent> second = [c.device newSharedEventWithHandle:handle];
    if (!first || !second) [c fail:@"newSharedEventWithHandle returned nil"];
    event.signaledValue = 7;
    [c value:@"first.sees-7" u64:first.signaledValue rule:WMExact()];
    [c value:@"second.sees-7" u64:second.signaledValue rule:WMExact()];
    second.signaledValue = 9;
    [c value:@"original.sees-9" u64:event.signaledValue rule:WMExact()];
    // A GPU signal on a recreated event is visible through the original.
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    [commandBuffer encodeSignalEvent:first value:12];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    [c value:@"original.sees-gpu-12" u64:event.signaledValue rule:WMExact()];
}

// MTLCommandBuffer.h, encodeSignalEvent:value: "This method may only be
// called if there is no current command encoder on the receiver."
WMREF_CASE(signalWithOpenEncoder, "event.signal-with-open-encoder", "event.errors",
           "Error path: encodeSignalEvent:value: while a blit encoder is open")
{
    id<MTLEvent> event = [c.device newEvent];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [c step:@"encode-signal"];
    [c call:@"encode-signal" rule:WMExact() block:^{
        [commandBuffer encodeSignalEvent:event value:1];
    }];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    [c value:@"status" u64:commandBuffer.status rule:WMNone(@"the documentation rules the call out")];
}

#pragma mark - Fences

static id<MTLBuffer> EventsUntrackedBuffer(WMCase *c, NSUInteger length) {
    id<MTLBuffer> buffer = [c.device newBufferWithLength:length
                                                 options:MTLResourceStorageModeShared |
                                                         MTLResourceHazardTrackingModeUntracked];
    if (!buffer) [c fail:@"untracked buffer creation returned nil"];
    memset(buffer.contents, 0, length);
    return buffer;
}

// MTLComputeCommandEncoder.h, waitForFence: "Prevent further GPU work until
// the fence is reached." Untracked resources are ordered only by fences
// (MTLResource.h, MTLResourceHazardTrackingModeUntracked).
WMREF_CASE(fenceAcrossEncoders, "fence.across-encoders", "fence.ordering",
           "Compute, blit and compute passes on untracked buffers ordered by one fence")
{
    id<MTLFence> fence = [c.device newFence];
    if (!fence) [c fail:@"newFence returned nil"];
    fence.label = @"pass-order";
    [c value:@"fence.label" string:fence.label rule:WMExact()];
    [c value:@"fence.device-is-creator" boolean:fence.device == c.device rule:WMExact()];
    enum { N = 1024 };
    id<MTLBuffer> first = EventsUntrackedBuffer(c, N * 4), second = EventsUntrackedBuffer(c, N * 4);
    uint32_t *words = first.contents;
    for (uint32_t i = 0; i < N; i++) words[i] = i;
    id<MTLComputePipelineState> add = [c computePipeline:@"cmd_add_value"];
    const uint32_t hundred = 100, thousand = 1000;
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];

    id<MTLComputeCommandEncoder> producer = [commandBuffer computeCommandEncoder];
    [producer setComputePipelineState:add];
    [producer setBuffer:first offset:0 atIndex:0];
    [producer setBytes:&hundred length:4 atIndex:1];
    [producer dispatchThreadgroups:MTLSizeMake(N / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [producer updateFence:fence];
    [producer endEncoding];

    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit waitForFence:fence];
    [blit copyFromBuffer:first sourceOffset:0 toBuffer:second destinationOffset:0 size:N * 4];
    [blit updateFence:fence];
    [blit endEncoding];

    id<MTLComputeCommandEncoder> consumer = [commandBuffer computeCommandEncoder];
    [consumer waitForFence:fence];
    [consumer setComputePipelineState:add];
    [consumer setBuffer:second offset:0 atIndex:0];
    [consumer setBytes:&thousand length:4 atIndex:1];
    [consumer dispatchThreadgroups:MTLSizeMake(N / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [consumer endEncoding];
    [c commitAndWait:commandBuffer];
    [c data:@"first" elem:WMElemU32 bytes:first.contents length:N * 4 rule:WMExact()];
    [c data:@"second" elem:WMElemU32 bytes:second.contents length:N * 4 rule:WMExact()];
}

// MTLComputeCommandEncoder.h, waitForFence: "The fence is evaluated at
// kernel submission to maintain global order and prevent deadlock." A pass
// waiting on a fence that only a later pass updates therefore does not wait
// for that later pass, and the command buffer completes.
WMREF_CASE(fenceWaitBeforeUpdate, "fence.wait-before-later-update", "fence.ordering",
           "A pass waits on a fence that only a later pass in the same command buffer updates")
{
    id<MTLFence> fence = [c.device newFence];
    id<MTLBuffer> a = EventsUntrackedBuffer(c, 256 * 4), b = EventsUntrackedBuffer(c, 256 * 4);
    id<MTLComputePipelineState> fill = [c computePipeline:@"cmd_fill"];
    const uint32_t one = 1, two = 2;
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> waiter = [commandBuffer computeCommandEncoder];
    [waiter waitForFence:fence];
    [waiter setComputePipelineState:fill];
    [waiter setBuffer:a offset:0 atIndex:0];
    [waiter setBytes:&one length:4 atIndex:1];
    [waiter dispatchThreadgroups:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [waiter endEncoding];
    id<MTLComputeCommandEncoder> updater = [commandBuffer computeCommandEncoder];
    [updater setComputePipelineState:fill];
    [updater setBuffer:b offset:0 atIndex:0];
    [updater setBytes:&two length:4 atIndex:1];
    [updater dispatchThreadgroups:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [updater updateFence:fence];
    [updater endEncoding];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    EventsSignalOnCompletion(commandBuffer, done);
    [commandBuffer commit];
    BOOL finished = EventsWait(done, 20);
    [c value:@"completed-within-20s" boolean:finished rule:WMExact()];
    if (!finished) return;
    [c value:@"status" u64:commandBuffer.status rule:WMExact()];
    [c data:@"a" elem:WMElemU32 bytes:a.contents length:a.length rule:WMExact()];
    [c data:@"b" elem:WMElemU32 bytes:b.contents length:b.length rule:WMExact()];
}

// MTLComputeCommandEncoder.h, waitForFence: "It is therefore illegal to
// wait on a fence after it has been updated in the same encoder."
WMREF_CASE(fenceUpdateThenWait, "fence.update-then-wait-same-encoder", "fence.errors",
           "Error path: waitForFence: after updateFence: on the same fence in one encoder")
{
    id<MTLFence> fence = [c.device newFence];
    id<MTLBuffer> a = EventsUntrackedBuffer(c, 256 * 4);
    id<MTLComputePipelineState> fill = [c computePipeline:@"cmd_fill"];
    const uint32_t one = 1;
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:fill];
    [encoder setBuffer:a offset:0 atIndex:0];
    [encoder setBytes:&one length:4 atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder updateFence:fence];
    [c step:@"wait-after-update"];
    [c call:@"wait-after-update" rule:WMExact() block:^{
        [encoder waitForFence:fence];
    }];
    [encoder endEncoding];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    EventsSignalOnCompletion(commandBuffer, done);
    [commandBuffer commit];
    BOOL finished = EventsWait(done, 20);
    WMRule *illegal = WMNone(@"MTLComputeCommandEncoder.h calls this use illegal and does not define its effect");
    [c value:@"completed-within-20s" boolean:finished rule:illegal];
    if (finished) [c value:@"status" u64:commandBuffer.status rule:illegal];
}

#pragma mark - Hazard tracking

// MTLResource.h, hazardTrackingMode: "This value can be either
// MTLHazardTrackingModeUntracked or MTLHazardTrackingModeTracked. Resources
// created from heaps are by default untracked, whereas resources created
// from the device are by default tracked." MTLHeap.h: "For heaps,
// MTLHazardTrackingModeDefault is treated as MTLHazardTrackingModeUntracked."
WMREF_CASE(hazardModes, "hazard.tracking-mode-readback", "hazard.modes",
           "hazardTrackingMode of device and heap resources for each hazard option")
{
    const MTLResourceOptions options[3] = {MTLResourceHazardTrackingModeDefault,
                                           MTLResourceHazardTrackingModeUntracked,
                                           MTLResourceHazardTrackingModeTracked};
    NSArray *names = @[@"default", @"untracked", @"tracked"];
    for (int i = 0; i < 3; i++) {
        id<MTLBuffer> buffer = [c.device newBufferWithLength:256 options:MTLResourceStorageModeShared | options[i]];
        if (!buffer) [c fail:@"buffer creation returned nil"];
        [c value:[NSString stringWithFormat:@"buffer.%@", names[i]] u64:buffer.hazardTrackingMode rule:WMExact()];
        MTLTextureDescriptor *descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:8 height:8
                                                           mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.hazardTrackingMode = (MTLHazardTrackingMode)(options[i] >> MTLResourceHazardTrackingModeShift);
        id<MTLTexture> texture = [c texture:descriptor];
        [c value:[NSString stringWithFormat:@"texture.%@", names[i]] u64:texture.hazardTrackingMode rule:WMExact()];

        MTLHeapDescriptor *heapDescriptor = [MTLHeapDescriptor new];
        heapDescriptor.size = 1 << 16;
        heapDescriptor.storageMode = MTLStorageModePrivate;
        heapDescriptor.hazardTrackingMode = descriptor.hazardTrackingMode;
        id<MTLHeap> heap = [c.device newHeapWithDescriptor:heapDescriptor];
        if (!heap) [c fail:@"heap creation returned nil"];
        [c value:[NSString stringWithFormat:@"heap.%@", names[i]] u64:heap.hazardTrackingMode rule:WMExact()];
        // No hazard bits: "All resources created from this heap shared the
        // same hazard tracking mode" (MTLHeap.h).
        id<MTLBuffer> sub = [heap newBufferWithLength:256 options:MTLResourceStorageModePrivate];
        if (!sub) [c fail:@"heap buffer creation returned nil"];
        [c value:[NSString stringWithFormat:@"heap-buffer.%@", names[i]] u64:sub.hazardTrackingMode rule:WMExact()];
    }
}

#pragma mark - Debug labels

// Labels are "A string to help identify this object" (MTLResource.h,
// MTLCommandBuffer.h, MTLCommandEncoder.h); debug groups and signposts
// annotate encoded work and do not change results.
WMREF_CASE(labelsAndGroups, "debug.labels-and-groups", "debug.labels",
           "Object labels read back; debug groups and signposts in a command buffer that completes")
{
    id<MTLCommandQueue> queue = [c.device newCommandQueue];
    [c value:@"queue.initial-label" string:queue.label rule:WMExact()];
    queue.label = @"queue-label";
    id<MTLBuffer> buffer = [c buffer:256 * 4];
    buffer.label = @"buffer-label";
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                                  width:4 height:4 mipmapped:NO];
    id<MTLTexture> texture = [c texture:td];
    texture.label = @"texture-label";
    id<MTLHeap> heap;
    {
        MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
        hd.size = 1 << 16;
        heap = [c.device newHeapWithDescriptor:hd];
        heap.label = @"heap-label";
    }
    id<MTLEvent> event = [c.device newEvent];
    event.label = @"event-label";
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    sd.label = @"sampler-label";
    id<MTLSamplerState> sampler = [c sampler:sd];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    [c value:@"command-buffer.initial-label" string:commandBuffer.label rule:WMExact()];
    commandBuffer.label = @"command-buffer-label";
    [commandBuffer pushDebugGroup:@"outer"];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [c value:@"encoder.initial-label" string:encoder.label rule:WMExact()];
    encoder.label = @"encoder-label";
    [encoder pushDebugGroup:@"inner"];
    [encoder insertDebugSignpost:@"signpost"];
    id<MTLComputePipelineState> add = [c computePipeline:@"cmd_add_value"];
    const uint32_t value = 4;
    [encoder setComputePipelineState:add];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&value length:4 atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder popDebugGroup];
    [c value:@"encoder.label" string:encoder.label rule:WMExact()];
    [encoder endEncoding];
    [commandBuffer popDebugGroup];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    [c value:@"queue.label" string:queue.label rule:WMExact()];
    [c value:@"buffer.label" string:buffer.label rule:WMExact()];
    [c value:@"texture.label" string:texture.label rule:WMExact()];
    [c value:@"heap.label" string:heap.label rule:WMExact()];
    [c value:@"event.label" string:event.label rule:WMExact()];
    [c value:@"sampler.label" string:sampler.label rule:WMExact()];
    [c value:@"command-buffer.label" string:commandBuffer.label rule:WMExact()];
    [c value:@"status" u64:commandBuffer.status rule:WMExact()];
    const uint32_t *words = buffer.contents;
    BOOL written = YES;
    for (int i = 0; i < 256; i++) written &= words[i] == 4;
    [c value:@"result-written" boolean:written rule:WMExact()];
}

#pragma mark - Counters

// MTLCounters.h: counter sets, sampling points, and resolveCounterRange:,
// which writes MTLCounterErrorValue for a sample that could not be taken.
// Timestamp values differ on every run; ordering properties are recorded.
WMREF_CASE(counterTimestamps, "counters.timestamps", "counters.timestamps",
           "Counter sets and sampling points of the device; stage-boundary timestamps of a compute pass")
{
    NSMutableArray<NSString *> *sets = [NSMutableArray array];
    id<MTLCounterSet> timestampSet = nil;
    for (id<MTLCounterSet> set in c.device.counterSets) {
        NSMutableArray *counters = [NSMutableArray array];
        for (id<MTLCounter> counter in set.counters) [counters addObject:counter.name];
        [counters sortUsingSelector:@selector(compare:)];
        [sets addObject:[NSString stringWithFormat:@"%@: %@", set.name, [counters componentsJoinedByString:@","]]];
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestampSet = set;
    }
    [sets sortUsingSelector:@selector(compare:)];
    [c value:@"counter-sets" strings:sets rule:WMDevice()];
    const MTLCounterSamplingPoint points[] = {
        MTLCounterSamplingPointAtStageBoundary, MTLCounterSamplingPointAtDrawBoundary,
        MTLCounterSamplingPointAtDispatchBoundary, MTLCounterSamplingPointAtTileDispatchBoundary,
        MTLCounterSamplingPointAtBlitBoundary,
    };
    NSArray *pointNames = @[@"stage", @"draw", @"dispatch", @"tile-dispatch", @"blit"];
    for (int i = 0; i < 5; i++)
        [c value:[NSString stringWithFormat:@"supports-sampling.%@", pointNames[i]]
            boolean:[c.device supportsCounterSampling:points[i]] rule:WMDevice()];
    MTLTimestamp cpu = 0, gpu = 0;
    [c.device sampleTimestamps:&cpu gpuTimestamp:&gpu];
    [c value:@"sample-timestamps.nonzero" boolean:cpu != 0 && gpu != 0 rule:WMDevice()];

    [c require:@"counter-set.timestamp" value:timestampSet != nil];
    [c require:@"sampling.stage-boundary" value:[c.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]];
    MTLCounterSampleBufferDescriptor *descriptor = [MTLCounterSampleBufferDescriptor new];
    descriptor.counterSet = timestampSet;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.sampleCount = 2;
    descriptor.label = @"timestamps";
    NSError *error = nil;
    id<MTLCounterSampleBuffer> samples = [c.device newCounterSampleBufferWithDescriptor:descriptor error:&error];
    if (!samples) [c fail:@"newCounterSampleBufferWithDescriptor: %@", error];
    [c value:@"sample-buffer.sample-count" u64:samples.sampleCount rule:WMExact()];
    MTLComputePassDescriptor *pass = [MTLComputePassDescriptor computePassDescriptor];
    pass.sampleBufferAttachments[0].sampleBuffer = samples;
    pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 0;
    pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 1;
    id<MTLBuffer> data = [c buffer:1024 * 4];
    id<MTLComputePipelineState> add = [c computePipeline:@"cmd_add_value"];
    const uint32_t value = 1;
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoderWithDescriptor:pass];
    [encoder setComputePipelineState:add];
    [encoder setBuffer:data offset:0 atIndex:0];
    [encoder setBytes:&value length:4 atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    [c commitAndWait:commandBuffer];
    NSData *resolved = [samples resolveCounterRange:NSMakeRange(0, 2)];
    if (resolved.length < 2 * sizeof(MTLCounterResultTimestamp)) [c fail:@"resolveCounterRange returned %lu bytes", (unsigned long)resolved.length];
    const MTLCounterResultTimestamp *stamps = resolved.bytes;
    [c value:@"resolved-length" u64:resolved.length rule:WMExact()];
    [c value:@"start.valid" boolean:stamps[0].timestamp != MTLCounterErrorValue && stamps[0].timestamp != 0 rule:WMExact()];
    [c value:@"end.valid" boolean:stamps[1].timestamp != MTLCounterErrorValue && stamps[1].timestamp != 0 rule:WMExact()];
    [c value:@"end-not-before-start" boolean:stamps[1].timestamp >= stamps[0].timestamp rule:WMExact()];
}

#pragma mark - IO command queue

static NSString *EventsWriteIOFile(WMCase *c, uint8_t *file, size_t length);

// MTLIOCommandBuffer.h: loadBuffer:offset:size:sourceHandle:sourceHandleOffset:
// "Encodes a command that loads from a handle and offset into a buffer and
// an offset"; status reports pending, cancelled, error or complete.
WMREF_CASE(ioLoadBuffer, "io.load-buffer", "io.queue",
           "MTLIOCommandQueue loads a file range into a buffer range")
{
    MTLIOCommandQueueDescriptor *descriptor = [MTLIOCommandQueueDescriptor new];
    NSError *error = nil;
    id<MTLIOCommandQueue> queue = [c.device newIOCommandQueueWithDescriptor:descriptor error:&error];
    [c require:@"io-command-queue" value:queue != nil];
    [c value:@"descriptor.type" i64:descriptor.type rule:WMExact()];
    [c value:@"descriptor.priority" i64:descriptor.priority rule:WMExact()];
    uint8_t file[8192];
    NSString *path = EventsWriteIOFile(c, file, sizeof file);
    id<MTLIOFileHandle> handle = [c.device newIOFileHandleWithURL:[NSURL fileURLWithPath:path] error:&error];
    if (!handle) {
        [[NSFileManager defaultManager] removeItemAtPath:path error:NULL];
        [c fail:@"newIOFileHandleWithURL: %@", error];
    }
    id<MTLBuffer> target = [c buffer:8192];
    memset(target.contents, 0xee, target.length);
    id<MTLIOCommandBuffer> commandBuffer = [queue commandBuffer];
    [commandBuffer loadBuffer:target offset:256 size:4096 sourceHandle:handle sourceHandleOffset:1024];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    [c value:@"status" i64:commandBuffer.status rule:WMExact()];
    [c error:@"error" value:commandBuffer.error rule:WMExact()];
    [c value:@"loaded-equals-file-range" boolean:memcmp((uint8_t *)target.contents + 256, file + 1024, 4096) == 0
        rule:WMExact()];
    [c data:@"target" elem:WMElemU8 bytes:target.contents length:target.length rule:WMExact()];

    [[NSFileManager defaultManager] removeItemAtPath:path error:NULL];
}

static NSString *EventsWriteIOFile(WMCase *c, uint8_t *file, size_t length) {
    WMPattern(file, length, 77);
    NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:
                      [NSString stringWithFormat:@"wmref-io-%d.bin", getpid()]];
    if (![[NSData dataWithBytes:file length:length] writeToFile:path atomically:YES])
        [c fail:@"cannot write %@", path];
    return path;
}

WMREF_CASE(ioLoadPastEnd, "io.load-past-end", "io.errors",
           "Error path: an IO load whose source range extends past the end of the file")
{
    MTLIOCommandQueueDescriptor *descriptor = [MTLIOCommandQueueDescriptor new];
    NSError *error = nil;
    id<MTLIOCommandQueue> queue = [c.device newIOCommandQueueWithDescriptor:descriptor error:&error];
    [c require:@"io-command-queue" value:queue != nil];
    uint8_t file[8192];
    NSString *path = EventsWriteIOFile(c, file, sizeof file);
    id<MTLIOFileHandle> handle = [c.device newIOFileHandleWithURL:[NSURL fileURLWithPath:path] error:&error];
    [[NSFileManager defaultManager] removeItemAtPath:path error:NULL];
    if (!handle) [c fail:@"newIOFileHandleWithURL: %@", error];
    id<MTLBuffer> target = [c buffer:1024];
    id<MTLIOCommandBuffer> commandBuffer = [queue commandBuffer];
    [c step:@"load"];
    [c call:@"load" rule:WMExact() block:^{
        [commandBuffer loadBuffer:target offset:0 size:1024 sourceHandle:handle sourceHandleOffset:7680];
    }];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    WMRule *undocumented = WMNone(@"MTLIOCommandBuffer.h does not state the result of reading past the end of a file");
    [c value:@"status" i64:commandBuffer.status rule:undocumented];
    [c error:@"error" value:commandBuffer.error rule:undocumented];
}
