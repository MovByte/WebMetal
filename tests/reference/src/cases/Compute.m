// Compute encoder cases: dispatch forms, builtins, threadgroup memory,
// indirect dispatch, inline constants, integer operations, conversions,
// atomics, stage-in and dispatch types.

#import "WMRefCase.h"

WMREF_CASE(computeThreadPositions, "compute.thread-positions", "compute.dispatch-threadgroups",
           "Builtin thread and threadgroup positions for dispatchThreadgroups")
{
    const uint32_t grid[4] = {12, 4, 2, 0};
    const uint32_t *gridWords = grid;
    MTLSize groups = MTLSizeMake(3, 2, 2), threads = MTLSizeMake(4, 2, 1);
    NSUInteger count = grid[0] * grid[1] * grid[2];
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"compute_positions"];
    id<MTLBuffer> out = [c buffer:count * 16 * sizeof(uint32_t)];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:gridWords length:4 * sizeof(uint32_t) atIndex:1];
        [e dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    }];
    [c data:@"records" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

WMREF_CASE(computeDispatchThreads, "compute.dispatch-threads-nonuniform", "compute.dispatch-threads",
           "dispatchThreads with a grid that is not a multiple of the threadgroup size")
{
    // Non-uniform threadgroup sizes: Metal feature set tables list them for
    // Apple4 and later and for Mac2.
    [c require:@"family.apple4-or-mac2"
         value:[c.device supportsFamily:MTLGPUFamilyApple4] || [c.device supportsFamily:(MTLGPUFamily)2002]];
    const uint32_t grid[4] = {7, 5, 3, 0};
    const uint32_t *gridWords = grid;
    NSUInteger slots = 8 * 6 * 4;
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"compute_positions"];
    id<MTLBuffer> out = [c buffer:slots * 16 * sizeof(uint32_t)];
    memset(out.contents, 0xee, out.length);
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:gridWords length:4 * sizeof(uint32_t) atIndex:1];
        [e dispatchThreads:MTLSizeMake(gridWords[0], gridWords[1], gridWords[2]) threadsPerThreadgroup:MTLSizeMake(4, 2, 2)];
    }];
    // Records past the grid's 105 threads keep the CPU fill: threads outside
    // the grid do not exist.
    [c data:@"records" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

WMREF_CASE(computeThreadgroupMemory, "compute.threadgroup-memory-length", "compute.threadgroup-memory",
           "Threadgroup memory sized by setThreadgroupMemoryLength:atIndex: with a barrier")
{
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"compute_threadgroup_mirror"];
    NSUInteger perGroup = 64, groups = 3;
    id<MTLBuffer> out = [c buffer:perGroup * groups * sizeof(uint32_t)];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:out offset:0 atIndex:0];
        [e setThreadgroupMemoryLength:perGroup * sizeof(uint32_t) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(perGroup, 1, 1)];
    }];
    [c data:@"mirrored" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

WMREF_CASE(computeIndirect, "compute.indirect-dispatch", "compute.indirect-dispatch",
           "dispatchThreadgroupsWithIndirectBuffer reading MTLDispatchThreadgroupsIndirectArguments at an offset")
{
    const uint32_t grid[4] = {6, 3, 1, 0};
    const uint32_t *gridWords = grid;
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"compute_positions"];
    id<MTLBuffer> out = [c buffer:grid[0] * grid[1] * 16 * sizeof(uint32_t)];
    // MTLDispatchThreadgroupsIndirectArguments is three uint32_t values.
    uint32_t arguments[8] = {0xdead, 0xbeef, 0xfeed, 0xface, 3, 3, 1, 0};
    id<MTLBuffer> indirect = [c bufferWithBytes:arguments length:sizeof arguments];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:gridWords length:4 * sizeof(uint32_t) atIndex:1];
        [e dispatchThreadgroupsWithIndirectBuffer:indirect indirectBufferOffset:16
                            threadsPerThreadgroup:MTLSizeMake(2, 1, 1)];
    }];
    [c data:@"records" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

WMREF_CASE(computeSetBytes, "compute.set-bytes", "compute.set-bytes",
           "setBytes:length:atIndex: with 4 KiB of inline constant data")
{
    uint32_t words[1024];
    WMPattern(words, sizeof words, 11);
    const uint32_t *wordsPointer = words;
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"compute_copy_constant"];
    id<MTLBuffer> out = [c buffer:sizeof words];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:wordsPointer length:1024 * sizeof(uint32_t) atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }];
    [c value:@"equal-to-input" boolean:memcmp(out.contents, words, sizeof words) == 0 rule:WMExact()];
    [c data:@"copied" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

WMREF_CASE(computeIntegerOps, "compute.integer-ops", "compute.integer-math",
           "MSL integer functions with defined results on generated inputs")
{
    NSUInteger n = 256;
    uint32_t a[256], b[256];
    WMPattern(a, sizeof a, 21);
    WMPattern(b, sizeof b, 22);
    // Include boundary values.
    const uint32_t edges[] = {0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0x80000001, 31, 32};
    for (NSUInteger i = 0; i < 8; i++)
        for (NSUInteger j = 0; j < 8; j++) {
            a[i * 8 + j] = edges[i];
            b[i * 8 + j] = edges[j];
        }
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"compute_integer_ops"];
    id<MTLBuffer> out = [c buffer:n * 32 * sizeof(uint32_t)];
    id<MTLBuffer> ba = [c bufferWithBytes:a length:sizeof a], bb = [c bufferWithBytes:b length:sizeof b];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBuffer:ba offset:0 atIndex:1];
        [e setBuffer:bb offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(n / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }];
    [c data:@"a" elem:WMElemU32 bytes:a length:sizeof a rule:WMExact()];
    [c data:@"b" elem:WMElemU32 bytes:b length:sizeof b rule:WMExact()];
    [c data:@"results" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

WMREF_CASE(computeConversions, "compute.conversions", "compute.conversions",
           "float to int (toward zero), float to half (ties to even), half to float (lossless)")
{
    // Inputs stay in range for every conversion and avoid NaN, infinity and
    // values whose half result would be subnormal, so section 8.6 defines
    // each result exactly.
    enum { N = 64 };
    float f[N];
    uint16_t h[N];
    const float directed[] = {
        0.0f, -0.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 1.0f + 0x1p-11f, 1.0f + 0x1p-11f + 0x1p-23f,
        1.0f + 3 * 0x1p-11f, 65504.0f, 65519.0f, 2048.0f + 1.0f, 2048.0f + 3.0f, 0x1p-14f,
        0x1p-14f + 0x1p-25f, 123456.75f, -98765.25f, 2147483520.0f,
    };
    size_t directedCount = sizeof directed / sizeof directed[0];
    uint32_t seed[N];
    WMPattern(seed, sizeof seed, 31);
    for (NSUInteger i = 0; i < N; i++) {
        if (i < directedCount) {
            f[i] = directed[i];
        } else {
            // Random normal floats with magnitude in [2^-14, 2^15).
            uint32_t bits = (seed[i] & 0x807fffff) | ((uint32_t)(113 + seed[i] % 29) << 23);
            memcpy(&f[i], &bits, sizeof bits);
        }
        // Normal half inputs (exponent field 1..30) and the two signed zeros.
        uint16_t bits = (uint16_t)(seed[i] & 0x83ff) | (uint16_t)((1 + (seed[i] >> 16) % 30) << 10);
        h[i] = i == 0 ? 0x0000 : i == 1 ? 0x8000 : bits;
    }
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"compute_conversions"];
    id<MTLBuffer> out = [c buffer:N * 4 * sizeof(uint32_t)];
    id<MTLBuffer> bf = [c bufferWithBytes:f length:sizeof f], bh = [c bufferWithBytes:h length:sizeof h];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBuffer:bf offset:0 atIndex:1];
        [e setBuffer:bh offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(N, 1, 1)];
    }];
    // CPU-computed expectations from the MSL 4.1 section 8.6 rules, so the
    // recorded results can be audited against the specification.
    uint32_t expected[N * 4];
    for (NSUInteger i = 0; i < N; i++) {
        expected[i * 4 + 0] = (uint32_t)(int32_t)truncf(f[i]);
        expected[i * 4 + 1] = WMHalfFromFloat(f[i]);
        float widened = (float)WMHalfToDouble(h[i]);
        memcpy(&expected[i * 4 + 2], &widened, 4);
        expected[i * 4 + 3] = f[i] >= 0 ? (uint32_t)truncf(f[i]) : 0;
    }
    [c data:@"float-inputs" elem:WMElemF32 bytes:f length:sizeof f rule:WMExact()];
    [c data:@"half-inputs" elem:WMElemU16 bytes:h length:sizeof h rule:WMExact()];
    [c data:@"results" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
    [c value:@"results-equal-specification" boolean:memcmp(expected, out.contents, sizeof expected) == 0
        rule:WMExact()];
}

WMREF_CASE(computeAtomics, "compute.atomics", "compute.atomics",
           "Device atomics from 1024 threads: final values and the set of fetched values")
{
    enum { N = 1024 };
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"compute_atomics"];
    uint32_t initial[8] = {0, 0, 0xffffffff, 0, 0xffffffff, 0, 1u << 20, 5};
    id<MTLBuffer> counters = [c bufferWithBytes:initial length:sizeof initial];
    id<MTLBuffer> fetched = [c buffer:N * sizeof(uint32_t)];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:counters offset:0 atIndex:0];
        [e setBuffer:fetched offset:0 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(N / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }];
    // The final values do not depend on the order of the atomic operations.
    [c data:@"counters" elem:WMElemU32 bytes:counters.contents length:counters.length rule:WMExact()];
    // The values returned by fetch_add are a permutation of 0..N-1; their
    // order depends on scheduling, so the sorted sequence is recorded.
    uint32_t sorted[N];
    memcpy(sorted, fetched.contents, sizeof sorted);
    qsort_b(sorted, N, sizeof(uint32_t), ^int(const void *x, const void *y) {
        uint32_t u = *(const uint32_t *)x, v = *(const uint32_t *)y;
        return u < v ? -1 : u > v;
    });
    [c data:@"fetch-add-results-sorted" elem:WMElemU32 bytes:sorted length:sizeof sorted rule:WMExact()];
}

WMREF_CASE(computeThreadgroupAtomics, "compute.threadgroup-atomics", "compute.atomics",
           "Threadgroup-memory atomics combined within each threadgroup")
{
    id<MTLComputePipelineState> pipeline = [c computePipeline:@"compute_threadgroup_atomics"];
    id<MTLBuffer> out = [c buffer:4 * 2 * sizeof(uint32_t)];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:out offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }];
    [c data:@"sum-and-max" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

WMREF_CASE(computeStageIn, "compute.stage-in-region", "compute.stage-in",
           "Compute stage_in fetched through MTLStageInputOutputDescriptor and setStageInRegion:")
{
    // Integer and float attribute formats only: the fetch passes their
    // values through unchanged.
    MTLStageInputOutputDescriptor *stage = [MTLStageInputOutputDescriptor stageInputOutputDescriptor];
    stage.attributes[0].format = MTLAttributeFormatUShort2;
    stage.attributes[0].offset = 0;
    stage.attributes[0].bufferIndex = 1;
    stage.attributes[1].format = MTLAttributeFormatFloat;
    stage.attributes[1].offset = 4;
    stage.attributes[1].bufferIndex = 1;
    stage.layouts[1].stride = 8;
    stage.layouts[1].stepFunction = MTLStepFunctionThreadPositionInGridX;
    stage.layouts[1].stepRate = 1;

    MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = [c function:@"compute_stage_in"];
    descriptor.stageInputDescriptor = stage;
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline =
        [c.device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone
                                             reflection:nil error:&error];
    if (!pipeline) [c fail:@"stage-in pipeline: %@", error];

    // 16 records of {ushort2, float}; the floats are finite normal values.
    uint32_t records[16 * 2];
    WMPattern(records, sizeof records, 41);
    for (int i = 0; i < 16; i++) records[i * 2 + 1] = (records[i * 2 + 1] & 0x807fffff) | (0x7eu << 23);
    id<MTLBuffer> input = [c bufferWithBytes:records length:sizeof records];
    id<MTLBuffer> out = [c buffer:6 * 2 * 4 * sizeof(uint32_t)];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBuffer:input offset:0 atIndex:1];
        [e setStageInRegion:MTLRegionMake2D(5, 0, 6, 2)];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(6, 2, 1)];
    }];
    [c data:@"input" elem:WMElemU32 bytes:records length:sizeof records rule:WMExact()];
    [c data:@"fetched" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

WMREF_CASE(computeConcurrentDispatch, "compute.dispatch-type-concurrent", "compute.dispatch-type",
           "Concurrent dispatch type with a buffer memory barrier between dependent dispatches")
{
    enum { N = 256 };
    id<MTLComputePipelineState> addOne = [c computePipeline:@"compute_add_one"];
    id<MTLComputePipelineState> scale = [c computePipeline:@"compute_scale"];
    uint32_t input[N];
    for (uint32_t i = 0; i < N; i++) input[i] = i * 3;
    id<MTLBuffer> data = [c bufferWithBytes:input length:sizeof input];
    id<MTLBuffer> out = [c buffer:sizeof input];
    const uint32_t factor = 5;
    id<MTLCommandBuffer> commandBuffer = [c commandBuffer];
    id<MTLComputeCommandEncoder> e = [commandBuffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [e setComputePipelineState:addOne];
    [e setBuffer:data offset:0 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(N / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [e setComputePipelineState:scale];
    [e setBuffer:out offset:0 atIndex:0];
    [e setBuffer:data offset:0 atIndex:1];
    [e setBytes:&factor length:sizeof factor atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(N / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [e endEncoding];
    [c commitAndWait:commandBuffer];
    [c value:@"dispatch-type" i64:MTLDispatchTypeConcurrent rule:WMExact()];
    [c data:@"result" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
}

WMREF_CASE(computePipelineProperties, "compute.pipeline-properties", "compute.pipeline-limits",
           "Compute pipeline limits with and without a descriptor maxTotalThreadsPerThreadgroup")
{
    id<MTLComputePipelineState> plain = [c computePipeline:@"compute_threadgroup_atomics"];
    [c value:@"max-total-threads" u64:plain.maxTotalThreadsPerThreadgroup rule:WMDevice()];
    [c value:@"thread-execution-width" u64:plain.threadExecutionWidth rule:WMDevice()];
    [c value:@"static-threadgroup-memory" u64:plain.staticThreadgroupMemoryLength rule:WMDevice()];

    MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = [c function:@"compute_threadgroup_atomics"];
    descriptor.maxTotalThreadsPerThreadgroup = 96;
    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
    NSError *error = nil;
    id<MTLComputePipelineState> limited =
        [c.device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone
                                             reflection:nil error:&error];
    if (!limited) [c fail:@"limited pipeline: %@", error];
    [c value:@"limited.max-total-threads" u64:limited.maxTotalThreadsPerThreadgroup rule:WMDevice()];
    [c value:@"limited.at-most-descriptor-limit" boolean:limited.maxTotalThreadsPerThreadgroup <= 96
        rule:WMExact()];
}
