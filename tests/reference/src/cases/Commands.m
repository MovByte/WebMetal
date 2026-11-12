// Command queue and command buffer cases: queue capacity, execution order,
// the status sequence, scheduled and completed handlers, timing properties,
// error reporting, reference retention and the error paths of the command
// buffer life cycle.

#import "WMRefCase.h"


static BOOL CommandsWait(dispatch_semaphore_t semaphore, double seconds) {
    return dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(seconds * NSEC_PER_SEC))) == 0;
}

// Encodes `cmd_add_value` over the first `count` words of `buffer`.
static void CommandsEncodeAdd(WMCase *c, id<MTLCommandBuffer> commandBuffer, id<MTLBuffer> buffer,
                              NSUInteger count, uint32_t value) {
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"cmd_add_value"];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&value length:sizeof value atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(count / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
}

static BOOL CommandsAllWords(id<MTLBuffer> buffer, NSUInteger count, uint32_t value) {
    const uint32_t *words = buffer.contents;
    for (NSUInteger i = 0; i < count; i++)
        if (words[i] != value) return NO;
    return YES;
}

// MTLCommandQueue documentation, commandBuffer: "Each command queue has a
// fixed number of command buffers for its lifetime. This method blocks the
// calling CPU thread when the queue doesn't have any free command buffers,
// and returns after the GPU finishes executing one."
WMREF_CASE(queueCapacity, "queue.max-command-buffer-count", "queue.capacity",
           "commandBuffer on a queue whose maxCommandBufferCount buffers are all outstanding")
{
    id<MTLCommandQueue> queue = [c.device newCommandQueueWithMaxCommandBufferCount:2];
    if (!queue) [c fail:@"newCommandQueueWithMaxCommandBufferCount:2 returned nil"];
    id<MTLCommandBuffer> first, second;
    @autoreleasepool {
        first = [queue commandBuffer];
        second = [queue commandBuffer];
    }
    if (!first || !second) [c fail:@"the first two command buffers of a two-buffer queue are nil"];

    dispatch_semaphore_t returned = dispatch_semaphore_create(0);
    __block BOOL thirdNil = NO;
    __block id<MTLCommandBuffer> third = nil;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        thirdNil = commandBuffer == nil;
        third = commandBuffer;
        dispatch_semaphore_signal(returned);
    });
    [c step:@"third-requested"];
    BOOL whileFull = CommandsWait(returned, 0.5);
    [c value:@"third.returned-while-two-outstanding" boolean:whileFull rule:WMExact()];
    if (!whileFull) {
        [first commit];
        [first waitUntilCompleted];
        [c step:@"first-completed"];
        BOOL afterCompletion = CommandsWait(returned, 5.0);
        [c value:@"third.returned-after-first-completed" boolean:afterCompletion rule:WMExact()];
        if (!afterCompletion) {
            first = nil;
            BOOL afterRelease = CommandsWait(returned, 5.0);
            [c value:@"third.returned-after-first-released" boolean:afterRelease rule:WMExact()];
            if (!afterRelease) [c fail:@"the third command buffer never returned"];
        }
    }
    [c value:@"third.is-nil" boolean:thirdNil rule:WMExact()];

    // A command buffer released without being committed: whether its slot
    // returns to the queue. The documentation states only the completion
    // case, so the observation is recorded without a comparison.
    id<MTLCommandQueue> single = [c.device newCommandQueueWithMaxCommandBufferCount:1];
    if (!single) [c fail:@"newCommandQueueWithMaxCommandBufferCount:1 returned nil"];
    @autoreleasepool {
        id<MTLCommandBuffer> dropped = [single commandBuffer];
        if (!dropped) [c fail:@"the first command buffer of a one-buffer queue is nil"];
        dropped = nil;
    }
    dispatch_semaphore_t again = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            (void)[single commandBuffer];
        }
        dispatch_semaphore_signal(again);
    });
    [c value:@"uncommitted-release.frees-slot" boolean:CommandsWait(again, 2.0)
        rule:WMNone(@"MTLCommandQueue documents only that a slot returns after the GPU finishes a command buffer")];
    [second commit];
    [second waitUntilCompleted];
    if (third) {
        [third commit];
        [third waitUntilCompleted];
    }
}

// MTLCommandBuffer.h, enqueue: "Append this command buffer to the end of its
// MTLCommandQueue." Command buffers of one queue run in queue order, so the
// buffer enqueued last writes last.
WMREF_CASE(enqueueOrder, "command-buffer.enqueue-order", "command-buffer.ordering",
           "Command buffers run in enqueue order when it differs from creation and commit order")
{
    id<MTLBuffer> target = [c buffer:256];
    id<MTLCommandBuffer> a = [c commandBuffer], b = [c commandBuffer];
    NSMutableArray<NSString *> *order = [NSMutableArray array];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    NSArray *buffers = @[a, b];
    const uint8_t values[2] = {0xa1, 0xb2};
    NSArray *names = @[@"a", @"b"];
    for (NSUInteger i = 0; i < 2; i++) {
        id<MTLCommandBuffer> commandBuffer = buffers[i];
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        [blit fillBuffer:target range:NSMakeRange(0, 256) value:values[i]];
        [blit endEncoding];
        NSString *name = names[i];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            @synchronized(order) {
                [order addObject:name];
            }
            dispatch_semaphore_signal(done);
        }];
    }
    [b enqueue];
    [a enqueue];
    [a commit];
    [b commit];
    if (!CommandsWait(done, 10) || !CommandsWait(done, 10)) [c fail:@"completed handlers did not run within 10 s"];
    [c value:@"a.status" u64:a.status rule:WMExact()];
    [c value:@"b.status" u64:b.status rule:WMExact()];
    [c data:@"target" elem:WMElemU8 bytes:target.contents length:256 rule:WMExact()];
    @synchronized(order) {
        [c value:@"completed-handler-order" strings:[order copy] rule:WMExact()];
    }
}

// MTLCommandBuffer.h, MTLCommandBufferStatus: not enqueued, enqueued,
// committed, scheduled, completed. MTLCommandBuffer documentation,
// addScheduledHandler: "The command buffer's status property may be equal to
// another (larger) value by the time your completion handler runs,
// including MTLCommandBufferStatus.completed."
WMREF_CASE(statusSequence, "command-buffer.status-sequence", "command-buffer.status",
           "status at creation, after enqueue, commit, waitUntilScheduled, waitUntilCompleted and in handlers")
{
    id<MTLBuffer> data = [c buffer:64 * sizeof(uint32_t)];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    [c value:@"created" u64:commandBuffer.status rule:WMExact()];
    CommandsEncodeAdd(c, commandBuffer, data, 64, 7);
    [c value:@"encoded" u64:commandBuffer.status rule:WMExact()];
    __block MTLCommandBufferStatus inScheduled = 0, inCompleted = 0;
    dispatch_semaphore_t handlers = dispatch_semaphore_create(0);
    [commandBuffer addScheduledHandler:^(id<MTLCommandBuffer> buffer) {
        inScheduled = buffer.status;
        dispatch_semaphore_signal(handlers);
    }];
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        inCompleted = buffer.status;
        dispatch_semaphore_signal(handlers);
    }];
    [commandBuffer enqueue];
    [c value:@"enqueued" u64:commandBuffer.status rule:WMExact()];
    [commandBuffer commit];
    // After commit the buffer is committed, then scheduled, then completed;
    // the device may advance it before the read, so the range is recorded.
    MTLCommandBufferStatus committed = commandBuffer.status;
    [c value:@"committed.committed-scheduled-or-completed"
        boolean:committed >= MTLCommandBufferStatusCommitted && committed <= MTLCommandBufferStatusCompleted
        rule:WMExact()];
    [commandBuffer waitUntilScheduled];
    // waitUntilScheduled returns once the buffer is scheduled; it may also
    // have completed by the read.
    MTLCommandBufferStatus scheduled = commandBuffer.status;
    [c value:@"after-wait-until-scheduled.scheduled-or-completed"
        boolean:scheduled == MTLCommandBufferStatusScheduled || scheduled == MTLCommandBufferStatusCompleted
        rule:WMExact()];
    [commandBuffer waitUntilCompleted];
    [c value:@"after-wait-until-completed" u64:commandBuffer.status rule:WMExact()];
    if (!CommandsWait(handlers, 10) || !CommandsWait(handlers, 10)) [c fail:@"handlers did not run within 10 s"];
    // addScheduledHandler documentation: the status "may be equal to another
    // (larger) value by the time your completion handler runs, including
    // MTLCommandBufferStatus.completed".
    [c value:@"in-scheduled-handler.scheduled-or-completed"
        boolean:inScheduled == MTLCommandBufferStatusScheduled || inScheduled == MTLCommandBufferStatusCompleted
        rule:WMExact()];
    [c value:@"in-completed-handler" u64:inCompleted rule:WMExact()];
    [c error:@"error" value:commandBuffer.error rule:WMExact()];
    [c value:@"result-written" boolean:CommandsAllWords(data, 64, 7) rule:WMExact()];

    // commit without an explicit enqueue.
    id<MTLCommandBuffer> implicit = [c commandBuffer];
    CommandsEncodeAdd(c, implicit, data, 64, 1);
    [implicit commit];
    MTLCommandBufferStatus implicitStatus = implicit.status;
    [c value:@"implicit.committed-scheduled-or-completed"
        boolean:implicitStatus >= MTLCommandBufferStatusCommitted && implicitStatus <= MTLCommandBufferStatusCompleted
        rule:WMExact()];
    [implicit waitUntilCompleted];
    [c value:@"implicit.after-wait-until-completed" u64:implicit.status rule:WMExact()];
}

WMREF_CASE(handlerOrder, "command-buffer.handler-order", "command-buffer.handlers",
           "Order of three scheduled and three completed handlers, and whether the waiting main thread runs one")
{
    id<MTLBuffer> data = [c buffer:64 * sizeof(uint32_t)];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    CommandsEncodeAdd(c, commandBuffer, data, 64, 3);
    NSMutableArray<NSString *> *sequence = [NSMutableArray array];
    __block BOOL anyOnMain = NO;
    dispatch_semaphore_t handlers = dispatch_semaphore_create(0);
    void (^note)(NSString *) = ^(NSString *name) {
        @synchronized(sequence) {
            [sequence addObject:name];
            if ([NSThread isMainThread]) anyOnMain = YES;
        }
        dispatch_semaphore_signal(handlers);
    };
    for (int i = 1; i <= 3; i++) {
        NSString *name = [NSString stringWithFormat:@"scheduled-%d", i];
        [commandBuffer addScheduledHandler:^(id<MTLCommandBuffer> buffer) { note(name); }];
    }
    for (int i = 1; i <= 3; i++) {
        NSString *name = [NSString stringWithFormat:@"completed-%d", i];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) { note(name); }];
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    for (int i = 0; i < 6; i++)
        if (!CommandsWait(handlers, 10)) [c fail:@"only %d of 6 handlers ran within 10 s", i];
    @synchronized(sequence) {
        // The documentation does not state an order among handlers; this is
        // the order Metal.framework's command buffer runs them in, which a
        // driver built on the framework's base classes inherits.
        [c value:@"sequence" strings:[sequence copy] rule:WMExact()];
        // The documentation says only that the device driver calls handlers
        // on the CPU. Which worker threads run them varies from run to run
        // and is not recorded; whether the waiting main thread runs one is.
        [c value:@"any-handler-on-main-thread" boolean:anyOnMain
            rule:WMNone(@"MTLCommandBuffer documentation does not specify the thread that runs handlers")];
    }
}

// MTLCommandBuffer.h: GPUStartTime "Returns zero if it has not started";
// GPUEndTime "Returns zero if CPU has not received completion notification".
// Raw times differ on every run, so ordering properties are recorded.
WMREF_CASE(timing, "command-buffer.timing", "command-buffer.timing",
           "GPU and kernel start and end times before commit and after completion, as ordering properties")
{
    id<MTLBuffer> data = [c buffer:256 * sizeof(uint32_t)];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    CommandsEncodeAdd(c, commandBuffer, data, 256, 1);
    [c value:@"before-commit.gpu-start-zero" boolean:commandBuffer.GPUStartTime == 0 rule:WMExact()];
    [c value:@"before-commit.gpu-end-zero" boolean:commandBuffer.GPUEndTime == 0 rule:WMExact()];
    [c value:@"before-commit.kernel-start-zero" boolean:commandBuffer.kernelStartTime == 0
        rule:WMNone(@"MTLCommandBuffer.h does not describe kernelStartTime before commit")];
    __block CFTimeInterval handlerStart = 0, handlerEnd = 0;
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        handlerStart = buffer.GPUStartTime;
        handlerEnd = buffer.GPUEndTime;
        dispatch_semaphore_signal(completed);
    }];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (!CommandsWait(completed, 10)) [c fail:@"completed handler did not run within 10 s"];
    CFTimeInterval gpuStart = commandBuffer.GPUStartTime, gpuEnd = commandBuffer.GPUEndTime;
    CFTimeInterval kernelStart = commandBuffer.kernelStartTime, kernelEnd = commandBuffer.kernelEndTime;
    [c value:@"completed.gpu-start-positive" boolean:gpuStart > 0 rule:WMExact()];
    [c value:@"completed.gpu-end-not-before-start" boolean:gpuEnd >= gpuStart rule:WMExact()];
    [c value:@"completed.kernel-start-positive" boolean:kernelStart > 0 rule:WMExact()];
    [c value:@"completed.kernel-end-not-before-start" boolean:kernelEnd >= kernelStart rule:WMExact()];
    [c value:@"completed-handler.gpu-times-set" boolean:handlerStart > 0 && handlerEnd >= handlerStart
        rule:WMExact()];
    // The documentation describes kernel times as CPU scheduling time and GPU
    // times as GPU execution time; their relative order is not stated.
    [c value:@"completed.kernel-start-not-after-gpu-start" boolean:kernelStart <= gpuStart
        rule:WMNone(@"the relation between kernel and GPU times is not documented")];
}

// MTLCommandBuffer.h, errorOptions and MTLCommandBufferDescriptor. A buffer
// that completes has no error; logs holds what shader logging produced,
// and these shaders do not log.
WMREF_CASE(successErrorAndLogs, "command-buffer.success-error-and-logs", "command-buffer.errors",
           "error, errorOptions and logs of a command buffer that completes")
{
    MTLCommandBufferDescriptor *descriptor = [MTLCommandBufferDescriptor new];
    [c value:@"descriptor-default.retained-references" boolean:descriptor.retainedReferences rule:WMExact()];
    [c value:@"descriptor-default.error-options" u64:descriptor.errorOptions rule:WMExact()];
    descriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    id<MTLCommandBuffer> commandBuffer = [c.queue commandBufferWithDescriptor:descriptor];
    if (!commandBuffer) [c fail:@"commandBufferWithDescriptor returned nil"];
    [c value:@"error-options" u64:commandBuffer.errorOptions rule:WMExact()];
    id<MTLBuffer> data = [c buffer:64 * sizeof(uint32_t)];
    CommandsEncodeAdd(c, commandBuffer, data, 64, 5);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    [c value:@"status" u64:commandBuffer.status rule:WMExact()];
    [c error:@"error" value:commandBuffer.error rule:WMExact()];
    NSUInteger logCount = 0;
    for (id log in commandBuffer.logs) {
        (void)log;
        logCount++;
    }
    [c value:@"log-count" u64:logCount rule:WMExact()];
    [c value:@"result-written" boolean:CommandsAllWords(data, 64, 5) rule:WMExact()];
}

// MTLResource.h, setPurgeableState: "Any access to the resource by either
// the CPU or device will be undefined" once the contents are discarded. The
// outcome of using an emptied buffer is therefore recorded for reference
// only; it is the device's answer, not a definition.
WMREF_CASE(purgeableEmptyUse, "command-buffer.purgeable-empty-resource", "command-buffer.errors",
           "A compute pass that reads and writes a buffer made purgeable-empty before commit")
{
    id<MTLBuffer> data = [c buffer:1024 * sizeof(uint32_t)];
    [data setPurgeableState:MTLPurgeableStateVolatile];
    [data setPurgeableState:MTLPurgeableStateEmpty];
    MTLCommandBufferDescriptor *descriptor = [MTLCommandBufferDescriptor new];
    descriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    id<MTLCommandBuffer> commandBuffer = [c.queue commandBufferWithDescriptor:descriptor];
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"cmd_add_value"];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"touch-emptied-buffer";
    const uint32_t value = 1;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:data offset:0 atIndex:0];
    [encoder setBytes:&value length:sizeof value atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        dispatch_semaphore_signal(completed);
    }];
    [commandBuffer commit];
    [c step:@"committed"];
    BOOL finished = CommandsWait(completed, 30);
    WMRule *undefined = WMNone(@"MTLResource.h: access to a resource whose contents were discarded is undefined");
    [c value:@"finished-within-30s" boolean:finished rule:undefined];
    if (!finished) return;
    [c value:@"status" u64:commandBuffer.status rule:undefined];
    [c error:@"error" value:commandBuffer.error rule:undefined];
    NSArray *infos = commandBuffer.error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
    NSMutableArray<NSString *> *states = [NSMutableArray array];
    for (id<MTLCommandBufferEncoderInfo> info in infos)
        [states addObject:[NSString stringWithFormat:@"%@:%ld", info.label, (long)info.errorState]];
    [c value:@"encoder-states" strings:states rule:undefined];
}

// MTLCommandQueue documentation: commandBuffer returns a buffer that
// "maintains strong references to all resources"; for
// commandBufferWithUnretainedReferences "This method sets the
// retainedReferences property to false".
WMREF_CASE(retainedReferences, "command-buffer.retained-references", "command-buffer.references",
           "retainedReferences for each way of creating a command buffer, and an unretained buffer in use")
{
    id<MTLCommandBuffer> retained = [c.queue commandBuffer];
    id<MTLCommandBuffer> unretained = [c.queue commandBufferWithUnretainedReferences];
    MTLCommandBufferDescriptor *descriptor = [MTLCommandBufferDescriptor new];
    descriptor.retainedReferences = NO;
    id<MTLCommandBuffer> described = [c.queue commandBufferWithDescriptor:descriptor];
    if (!retained || !unretained || !described) [c fail:@"command buffer creation returned nil"];
    [c value:@"command-buffer.retained-references" boolean:retained.retainedReferences rule:WMExact()];
    [c value:@"unretained.retained-references" boolean:unretained.retainedReferences rule:WMExact()];
    [c value:@"descriptor-no.retained-references" boolean:described.retainedReferences rule:WMExact()];
    [c value:@"command-queue-is-creator" boolean:retained.commandQueue == c.queue rule:WMExact()];
    [c value:@"device-is-creator" boolean:retained.device == c.device rule:WMExact()];

    // The caller keeps every object the commands use alive (the buffer and
    // the pipeline state); an unretained buffer then runs normally.
    id<MTLBuffer> data = [c buffer:64 * sizeof(uint32_t)];
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"cmd_add_value"];
    const uint32_t nine = 9;
    id<MTLComputeCommandEncoder> encoder = [unretained computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:data offset:0 atIndex:0];
    [encoder setBytes:&nine length:sizeof nine atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    [unretained commit];
    [unretained waitUntilCompleted];
    [c value:@"unretained.status" u64:unretained.status rule:WMExact()];
    [c value:@"unretained.result-written" boolean:CommandsAllWords(data, 64, 9) rule:WMExact()];
    [retained commit];
    [described commit];
    [retained waitUntilCompleted];
    [described waitUntilCompleted];
    (void)pipeline;
}

#pragma mark - Error paths

// Each case below makes one call the documentation rules out. The recorded
// result is what Metal does: an Objective-C exception, an assertion that
// ends the process (recorded by the runner as the outcome), or a return.

WMREF_CASE(commitTwice, "command-buffer.commit-twice", "command-buffer.errors",
           "Error path: commit on a command buffer that is already committed and completed")
{
    id<MTLBuffer> data = [c buffer:64 * sizeof(uint32_t)];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    CommandsEncodeAdd(c, commandBuffer, data, 64, 1);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    [c step:@"second-commit"];
    [c call:@"second-commit" rule:WMExact() block:^{
        [commandBuffer commit];
    }];
    [c value:@"status-after" u64:commandBuffer.status
        rule:WMNone(@"Metal does not define the result of a second commit")];
}

// addCompletedHandler documentation: "You can only call this method before
// calling the command buffer's commit() method."
WMREF_CASE(completedHandlerAfterCommit, "command-buffer.completed-handler-after-commit", "command-buffer.errors",
           "Error path: addCompletedHandler: after commit")
{
    id<MTLBuffer> data = [c buffer:64 * sizeof(uint32_t)];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    CommandsEncodeAdd(c, commandBuffer, data, 64, 1);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    dispatch_semaphore_t ran = dispatch_semaphore_create(0);
    [c step:@"add-completed-handler"];
    [c call:@"add-completed-handler" rule:WMExact() block:^{
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            dispatch_semaphore_signal(ran);
        }];
    }];
    [c value:@"handler-ran-within-2s" boolean:CommandsWait(ran, 2)
        rule:WMNone(@"the documentation rules the call out and does not define its effect")];
}

// addScheduledHandler documentation: "You can only call this method before
// calling the command buffer's commit() method."
WMREF_CASE(scheduledHandlerAfterCommit, "command-buffer.scheduled-handler-after-commit", "command-buffer.errors",
           "Error path: addScheduledHandler: after the buffer completed")
{
    id<MTLBuffer> data = [c buffer:64 * sizeof(uint32_t)];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    CommandsEncodeAdd(c, commandBuffer, data, 64, 1);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    dispatch_semaphore_t ran = dispatch_semaphore_create(0);
    [c step:@"add-scheduled-handler"];
    [c call:@"add-scheduled-handler" rule:WMExact() block:^{
        [commandBuffer addScheduledHandler:^(id<MTLCommandBuffer> buffer) {
            dispatch_semaphore_signal(ran);
        }];
    }];
    [c value:@"handler-ran-within-2s" boolean:CommandsWait(ran, 2)
        rule:WMNone(@"the documentation rules the call out and does not define its effect")];
}

WMREF_CASE(commitWithActiveEncoder, "command-buffer.commit-with-active-encoder", "command-buffer.errors",
           "Error path: commit while a compute encoder has not ended encoding")
{
    id<MTLBuffer> data = [c buffer:64 * sizeof(uint32_t)];
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"cmd_add_value"];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    const uint32_t value = 1;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:data offset:0 atIndex:0];
    [encoder setBytes:&value length:sizeof value atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [c step:@"commit"];
    [c call:@"commit" rule:WMExact() block:^{
        [commandBuffer commit];
    }];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        [commandBuffer waitUntilCompleted];
        dispatch_semaphore_signal(completed);
    });
    WMRule *undefined = WMNone(@"committing with an open encoder is outside the documented use");
    BOOL finished = CommandsWait(completed, 10);
    [c value:@"completed-within-10s" boolean:finished rule:undefined];
    if (finished) [c value:@"status" u64:commandBuffer.status rule:undefined];
}

WMREF_CASE(encoderReleasedOpen, "command-buffer.encoder-released-without-end-encoding", "command-buffer.errors",
           "Error path: an encoder released without endEncoding")
{
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    [c step:@"release-encoder"];
    [c call:@"release-encoder" rule:WMExact() block:^{
        @autoreleasepool {
            id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
            encoder.label = @"never-ended";
            encoder = nil;
        }
    }];
    [c step:@"commit"];
    [c call:@"commit" rule:WMExact() block:^{
        [commandBuffer commit];
    }];
}

WMREF_CASE(enqueueAfterCommit, "command-buffer.enqueue-after-commit", "command-buffer.errors",
           "Error path: enqueue on a committed command buffer")
{
    id<MTLBuffer> data = [c buffer:64 * sizeof(uint32_t)];
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    CommandsEncodeAdd(c, commandBuffer, data, 64, 1);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    [c step:@"enqueue"];
    [c call:@"enqueue" rule:WMExact() block:^{
        [commandBuffer enqueue];
    }];
    [c value:@"status-after" u64:commandBuffer.status
        rule:WMNone(@"Metal does not define enqueue after commit")];
}

WMREF_CASE(secondEncoderWhileOpen, "command-buffer.encoder-while-encoder-open", "command-buffer.errors",
           "Error path: a blit encoder requested while a compute encoder is open")
{
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> compute = [commandBuffer computeCommandEncoder];
    [c step:@"blit-encoder"];
    __block BOOL isNil = NO;
    [c call:@"blit-encoder" rule:WMExact() block:^{
        isNil = [commandBuffer blitCommandEncoder] == nil;
    }];
    [c value:@"blit-encoder-is-nil" boolean:isNil rule:WMNone(@"Metal does not define a second open encoder")];
    [compute endEncoding];
}
