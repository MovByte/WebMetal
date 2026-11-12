// Library and function cases: source compilation (synchronous and
// asynchronous), compile errors and warnings, compile options with defined
// effects, libraries from data and URLs, the default library, function
// lookup, function constants, function descriptors and function
// reflection.

#import "WMRefCase.h"
#import "LibrariesSupport.h"


static NSString *const WMSquareSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "#ifndef VALUE_OFFSET\n"
     "#define VALUE_OFFSET 0u\n"
     "#endif\n"
     "kernel void src_square(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]])\n"
     "{\n"
     "    out[i] = i * i + VALUE_OFFSET;\n"
     "}\n"
     "kernel void src_version(device uint *out [[buffer(0)]])\n"
     "{\n"
     "    out[0] = __METAL_VERSION__;\n"
     "#if defined(__METAL_MACOS__)\n"
     "    out[1] = 1u;\n"
     "#elif defined(__METAL_IOS__)\n"
     "    out[1] = 2u;\n"
     "#else\n"
     "    out[1] = 3u;\n"
     "#endif\n"
     "}\n";

// Floating-point special values read from a buffer so that they cannot be
// folded at compile time.
static NSString *const WMSpecialValuesSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void src_special(device uint *out [[buffer(0)]], device const float *in [[buffer(1)]])\n"
     "{\n"
     "    float nan_value = in[0], inf_value = in[1], one = in[2], zero = in[3];\n"
     "    out[0] = isnan(nan_value) ? 1u : 0u;\n"
     "    out[1] = isinf(inf_value) ? 1u : 0u;\n"
     "    out[2] = isnan(inf_value * zero) ? 1u : 0u;\n"
     "    out[3] = as_type<uint>(one / zero);\n"
     "    out[4] = as_type<uint>(one * -zero);\n"
     "    out[5] = isnan(inf_value - inf_value) ? 1u : 0u;\n"
     "    out[6] = (nan_value == nan_value) ? 1u : 0u;\n"
     "    out[7] = as_type<uint>(-inf_value);\n"
     "}\n";

NSString *WMRefStableText(NSString *text) {
    NSString *temporary = NSTemporaryDirectory();
    NSString *stable = text;
    for (NSString *prefix in @[temporary, [@"/private" stringByAppendingString:temporary]])
        stable = [stable stringByReplacingOccurrencesOfString:prefix withString:@"$TMPDIR/"];
    NSRegularExpression *address = [NSRegularExpression regularExpressionWithPattern:@"0x[0-9a-fA-F]+" options:0
                                                                               error:NULL];
    stable = [address stringByReplacingMatchesInString:stable options:0 range:NSMakeRange(0, stable.length)
                                          withTemplate:@"0x"];
    NSRegularExpression *pid = [NSRegularExpression regularExpressionWithPattern:@"wmref-[0-9]+-" options:0
                                                                           error:NULL];
    return [pid stringByReplacingMatchesInString:stable options:0 range:NSMakeRange(0, stable.length)
                                    withTemplate:@"wmref-"];
}

NSError *WMRefStableError(NSError *error) {
    if (!error) return nil;
    return [NSError errorWithDomain:error.domain code:error.code
                           userInfo:@{NSLocalizedDescriptionKey: WMRefStableText(error.localizedDescription ?: @"")}];
}

WMRule *WMRefDriverErrorRule(void) {
    return WMDevice();
}

BOOL WMRefWait(dispatch_semaphore_t semaphore, double seconds) {
    return dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(seconds * NSEC_PER_SEC))) == 0;
}

NSArray<NSString *> *WMRefSortedNames(NSArray<NSString *> *names) {
    return [names sortedArrayUsingSelector:@selector(compare:)];
}

// Runs a kernel with one buffer argument of `words` 32-bit words over
// `threads` threads and returns the buffer contents.
NSData *WMRefRunWordsKernel(WMCase *c, id<MTLFunction> function, NSUInteger words, NSUInteger threads) {
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [c.device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) [c fail:@"pipeline for %@: %@", function.name, error];
    id<MTLBuffer> out = [c buffer:words * sizeof(uint32_t)];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:out offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }];
    return [NSData dataWithBytes:out.contents length:out.length];
}

static id<MTLFunction> WMSourceFunction(WMCase *c, id<MTLLibrary> library, NSString *name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) [c fail:@"function %@ missing from a source library", name];
    return function;
}

WMREF_CASE(librarySourceSync, "library.source-sync", "library.source",
           "newLibraryWithSource:options:error: compiles, lists functions and runs a kernel")
{
    NSError *error = nil;
    id<MTLLibrary> library = [c.device newLibraryWithSource:WMSquareSource options:nil error:&error];
    [c value:@"library-nil" boolean:library == nil rule:WMExact()];
    [c error:@"error" value:error rule:WMExact()];
    if (!library) [c fail:@"source library did not compile"];
    [c value:@"function-names" strings:WMRefSortedNames(library.functionNames) rule:WMExact()];
    [c value:@"type" i64:library.type rule:WMExact()];
    [c value:@"install-name-nil" boolean:library.installName == nil rule:WMExact()];
    [c value:@"label-nil" boolean:library.label == nil rule:WMExact()];
    NSData *squares = WMRefRunWordsKernel(c, WMSourceFunction(c, library, @"src_square"), 16, 16);
    [c data:@"squares" elem:WMElemU32 bytes:squares.bytes length:squares.length rule:WMExact()];
}

WMREF_CASE(librarySourceAsync, "library.source-async", "library.source",
           "newLibraryWithSource:options:completionHandler: for a valid and an invalid source")
{
    __block NSUInteger calls = 0;
    __block id<MTLLibrary> library = nil;
    __block NSError *error = nil;
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [c.device newLibraryWithSource:WMSquareSource options:nil
                 completionHandler:^(id<MTLLibrary> result, NSError *resultError) {
        calls++;
        library = result;
        error = resultError;
        dispatch_semaphore_signal(done);
    }];
    if (!WMRefWait(done, 120)) [c fail:@"completion handler not called within 120 s"];
    // A late second call would land here.
    [NSThread sleepForTimeInterval:0.2];
    [c value:@"valid.handler-calls" u64:calls rule:WMExact()];
    [c value:@"valid.library-nil" boolean:library == nil rule:WMExact()];
    [c error:@"valid.error" value:error rule:WMExact()];
    if (library) {
        NSData *squares = WMRefRunWordsKernel(c, WMSourceFunction(c, library, @"src_square"), 8, 8);
        [c data:@"valid.squares" elem:WMElemU32 bytes:squares.bytes length:squares.length rule:WMExact()];
    }

    calls = 0;
    library = nil;
    error = nil;
    [c.device newLibraryWithSource:@"kernel void broken( {" options:nil
                 completionHandler:^(id<MTLLibrary> result, NSError *resultError) {
        calls++;
        library = result;
        error = resultError;
        dispatch_semaphore_signal(done);
    }];
    if (!WMRefWait(done, 120)) [c fail:@"completion handler not called within 120 s"];
    [NSThread sleepForTimeInterval:0.2];
    [c value:@"invalid.handler-calls" u64:calls rule:WMExact()];
    [c value:@"invalid.library-nil" boolean:library == nil rule:WMExact()];
    [c error:@"invalid.error" value:error rule:WMExact()];
}

WMREF_CASE(librarySourceErrors, "library.source-compile-error", "library.source-errors",
           "Compile failures and warnings from newLibraryWithSource:options:error:")
{
    NSError *error = nil;
    id<MTLLibrary> library = [c.device newLibraryWithSource:@"kernel void broken( {" options:nil error:&error];
    [c value:@"syntax.library-nil" boolean:library == nil rule:WMExact()];
    [c error:@"syntax.error" value:error rule:WMExact()];

    error = nil;
    library = [c.device newLibraryWithSource:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
         "kernel void k(device uint *o [[buffer(0)]]) { o[0] = undeclared_name; }\n"
                                     options:nil error:&error];
    [c value:@"undeclared.library-nil" boolean:library == nil rule:WMExact()];
    [c error:@"undeclared.error" value:error rule:WMExact()];

    // A source that compiles with a diagnostic warning (unused variable).
    error = nil;
    library = [c.device newLibraryWithSource:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
         "kernel void k(device uint *o [[buffer(0)]]) { uint unused_value = 3; o[0] = 1u; }\n"
                                     options:nil error:&error];
    [c value:@"warning.library-nil" boolean:library == nil rule:WMExact()];
    [c error:@"warning.error" value:error rule:WMExact()];

    // An empty source has no functions.
    error = nil;
    library = [c.device newLibraryWithSource:@"" options:nil error:&error];
    [c value:@"empty.library-nil" boolean:library == nil rule:WMExact()];
    [c error:@"empty.error" value:error rule:WMExact()];
    if (library) [c value:@"empty.function-count" u64:library.functionNames.count rule:WMExact()];
}

WMREF_CASE(libraryCompileOptions, "library.compile-options", "library.compile-options",
           "MTLCompileOptions defaults, preprocessor macros and language versions")
{
    // Defaults stated in MTLLibrary.h: preprocessorMacros nil, libraryType
    // executable, preserveInvariance false, mathFloatingPointFunctions fast.
    MTLCompileOptions *defaults = [MTLCompileOptions new];
    [c value:@"default.preprocessor-macros-nil" boolean:defaults.preprocessorMacros == nil rule:WMExact()];
    [c value:@"default.library-type" i64:defaults.libraryType rule:WMExact()];
    [c value:@"default.preserve-invariance" boolean:defaults.preserveInvariance rule:WMExact()];
    if (@available(macOS 15.0, iOS 18.0, *)) {
        [c value:@"default.math-floating-point-functions" i64:defaults.mathFloatingPointFunctions rule:WMExact()];
        [c value:@"default.math-mode" i64:defaults.mathMode
            rule:WMNone(@"MTLLibrary.h: the default depends on the language standard version")];
    }
    [c value:@"default.language-version" u64:defaults.languageVersion
        rule:WMNone(@"MTLLibrary.h does not state the default language version")];

    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.preprocessorMacros = @{@"VALUE_OFFSET": @"7u"};
    id<MTLLibrary> library = [c.device newLibraryWithSource:WMSquareSource options:options error:&error];
    if (!library) [c fail:@"macro library: %@", error];
    NSData *data = WMRefRunWordsKernel(c, WMSourceFunction(c, library, @"src_square"), 4, 4);
    [c data:@"macro-string.squares" elem:WMElemU32 bytes:data.bytes length:data.length rule:WMExact()];

    options.preprocessorMacros = @{@"VALUE_OFFSET": @9};
    library = [c.device newLibraryWithSource:WMSquareSource options:options error:&error];
    if (!library) [c fail:@"numeric macro library: %@", error];
    data = WMRefRunWordsKernel(c, WMSourceFunction(c, library, @"src_square"), 4, 4);
    [c data:@"macro-number.squares" elem:WMElemU32 bytes:data.bytes length:data.length rule:WMExact()];

    // __METAL_VERSION__ is MajorMinorPatch (MSL 4.1 section 1.6.2).
    struct { MTLLanguageVersion version; const char *name; } versions[] = {
        {MTLLanguageVersion2_0, "2.0"}, {MTLLanguageVersion2_4, "2.4"}, {MTLLanguageVersion3_0, "3.0"},
    };
    for (size_t i = 0; i < sizeof versions / sizeof versions[0]; i++) {
        MTLCompileOptions *versioned = [MTLCompileOptions new];
        versioned.languageVersion = versions[i].version;
        error = nil;
        id<MTLLibrary> lib = [c.device newLibraryWithSource:WMSquareSource options:versioned error:&error];
        NSString *base = [NSString stringWithFormat:@"language-%s", versions[i].name];
        [c value:[base stringByAppendingString:@".library-nil"] boolean:lib == nil rule:WMExact()];
        if (!lib) continue;
        data = WMRefRunWordsKernel(c, WMSourceFunction(c, lib, @"src_version"), 2, 1);
        [c data:[base stringByAppendingString:@".version-and-platform"] elem:WMElemU32 bytes:data.bytes
         length:data.length rule:WMExact()];
    }
}

WMREF_CASE(libraryMathMode, "library.math-mode", "library.compile-options",
           "IEEE special values under safe math (defined) and fast math (undefined)")
{
    const float inputs[4] = {NAN, INFINITY, 1.0f, 0.0f};
    id<MTLBuffer> in = [c bufferWithBytes:inputs length:sizeof inputs];
    for (int fast = 0; fast < 2; fast++) {
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, iOS 18.0, *)) {
            options.mathMode = fast ? MTLMathModeFast : MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = fast ? YES : NO;
#pragma clang diagnostic pop
        }
        NSError *error = nil;
        id<MTLLibrary> library = [c.device newLibraryWithSource:WMSpecialValuesSource options:options error:&error];
        if (!library) [c fail:@"special-values library: %@", error];
        id<MTLComputePipelineState> pipeline =
            [c.device newComputePipelineStateWithFunction:WMSourceFunction(c, library, @"src_special") error:&error];
        if (!pipeline) [c fail:@"special-values pipeline: %@", error];
        id<MTLBuffer> out = [c buffer:8 * sizeof(uint32_t)];
        [c compute:^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pipeline];
            [e setBuffer:out offset:0 atIndex:0];
            [e setBuffer:in offset:0 atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }];
        if (fast) {
            [c data:@"fast.results" elem:WMElemU32 bytes:out.contents length:out.length
               rule:WMNone(@"MSL 4.1 section 8.1: with fast math the handling of NaN and INF is undefined")];
        } else {
            // Safe math disables optimizations that affect results; NaN and
            // INF are supported with fast math disabled (MSL 4.1 sections
            // 1.6.3 and 8.1) and the operations follow IEEE 754.
            [c data:@"safe.results" elem:WMElemU32 bytes:out.contents length:out.length rule:WMExact()];
        }
    }
}

WMREF_CASE(libraryFromData, "library.data", "library.data",
           "newLibraryWithData:error: and newLibraryWithURL:error: with the harness metallib")
{
    NSURL *url = c.libraryURL;
    NSData *bytes = [NSData dataWithContentsOfURL:url];
    if (!bytes) [c fail:@"cannot read the harness library"];
    dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *error = nil;
    id<MTLLibrary> library = [c.device newLibraryWithData:data error:&error];
    [c value:@"data.library-nil" boolean:library == nil rule:WMExact()];
    [c error:@"data.error" value:error rule:WMExact()];
    if (!library) [c fail:@"harness library from data: %@", error];
    [c value:@"data.same-function-names" boolean:[WMRefSortedNames(library.functionNames)
                                                    isEqualToArray:WMRefSortedNames(c.library.functionNames)]
        rule:WMExact()];
    NSData *marker = WMRefRunWordsKernel(c, WMSourceFunction(c, library, @"pip_marker"), 4, 4);
    [c data:@"data.marker" elem:WMElemU32 bytes:marker.bytes length:marker.length rule:WMExact()];

    error = nil;
    id<MTLLibrary> fromURL = [c.device newLibraryWithURL:url error:&error];
    [c value:@"url.library-nil" boolean:fromURL == nil rule:WMExact()];
    [c error:@"url.error" value:error rule:WMExact()];
    [c value:@"url.function-has-marker" boolean:[fromURL.functionNames containsObject:@"pip_marker"]
        rule:WMExact()];
}

WMREF_CASE(libraryInvalidData, "library.data-invalid", "library.data",
           "newLibraryWithData:error: with bytes that are not a metallib")
{
    uint8_t garbage[256];
    WMPattern(garbage, sizeof garbage, 51);
    dispatch_data_t data = dispatch_data_create(garbage, sizeof garbage, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *error = nil;
    id<MTLLibrary> library = [c.device newLibraryWithData:data error:&error];
    [c value:@"library-nil" boolean:library == nil rule:WMExact()];
    [c error:@"error" value:error rule:WMExact()];
}

WMREF_CASE(libraryMissingURL, "library.url-missing", "library.data",
           "newLibraryWithURL:error: for a file that does not exist")
{
    NSURL *url = [NSURL fileURLWithPath:@"/nonexistent/wmref-missing.metallib"];
    NSError *error = nil;
    id<MTLLibrary> library = [c.device newLibraryWithURL:url error:&error];
    [c value:@"library-nil" boolean:library == nil rule:WMExact()];
    [c error:@"error" value:error rule:WMExact()];
}

WMREF_CASE(libraryDefault, "library.default-absent", "library.default",
           "newDefaultLibrary and newDefaultLibraryWithBundle:error: in a tool without default.metallib")
{
    // The harness is a command-line tool; its main bundle has no
    // default.metallib resource.
    [c require:@"main-bundle-without-default-metallib"
         value:[NSBundle.mainBundle URLForResource:@"default" withExtension:@"metallib"] == nil];
    id<MTLLibrary> library = [c.device newDefaultLibrary];
    [c value:@"default.library-nil" boolean:library == nil rule:WMExact()];
    NSError *error = nil;
    library = [c.device newDefaultLibraryWithBundle:NSBundle.mainBundle error:&error];
    [c value:@"bundle.library-nil" boolean:library == nil rule:WMExact()];
    [c error:@"bundle.error" value:error rule:WMExact()];
}

WMREF_CASE(functionMissing, "function.missing", "function.lookup",
           "Looking up a function name that the library does not contain")
{
    id<MTLFunction> function = [c.library newFunctionWithName:@"pip_no_such_function"];
    [c value:@"by-name.nil" boolean:function == nil rule:WMExact()];
    NSError *error = nil;
    function = [c.library newFunctionWithName:@"pip_no_such_function"
                               constantValues:[MTLFunctionConstantValues new] error:&error];
    [c value:@"with-constants.nil" boolean:function == nil rule:WMExact()];
    [c error:@"with-constants.error" value:error rule:WMExact()];
    MTLFunctionDescriptor *descriptor = [MTLFunctionDescriptor functionDescriptor];
    descriptor.name = @"pip_no_such_function";
    error = nil;
    function = [c.library newFunctionWithDescriptor:descriptor error:&error];
    [c value:@"descriptor.nil" boolean:function == nil rule:WMExact()];
    [c error:@"descriptor.error" value:error rule:WMExact()];
}

static NSArray<NSString *> *WMDescribeConstants(NSDictionary<NSString *, MTLFunctionConstant *> *constants) {
    NSMutableArray<NSString *> *entries = [NSMutableArray array];
    for (NSString *key in constants) {
        MTLFunctionConstant *constant = constants[key];
        [entries addObject:[NSString stringWithFormat:@"%@ name=%@ type=%lu index=%lu required=%d", key,
                            constant.name, (unsigned long)constant.type, (unsigned long)constant.index,
                            constant.required]];
    }
    return WMRefSortedNames(entries);
}

static MTLFunctionConstantValues *WMBaseConstants(BOOL flag) {
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    const uint32_t a = 0x12345678;
    const bool flagValue = flag;
    const float scale = 1.5f;
    const uint16_t half = 0x3c00; // 1.0 in binary16
    const int32_t vector[2] = {-3, 7};
    const uint16_t named = 0xbeef;
    [values setConstantValue:&a type:MTLDataTypeUInt atIndex:0];
    [values setConstantValue:&flagValue type:MTLDataTypeBool atIndex:1];
    [values setConstantValue:&scale type:MTLDataTypeFloat atIndex:2];
    [values setConstantValue:&half type:MTLDataTypeHalf atIndex:4];
    [values setConstantValue:vector type:MTLDataTypeInt2 atIndex:5];
    [values setConstantValue:&named type:MTLDataTypeUShort withName:@"pip_fc_named"];
    return values;
}

static NSData *WMRunConstants(WMCase *c, id<MTLFunction> function, BOOL bindExtra, NSData **extraOut) {
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [c.device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) [c fail:@"constants pipeline: %@", error];
    id<MTLBuffer> out = [c buffer:8 * sizeof(uint32_t)];
    id<MTLBuffer> extra = [c buffer:sizeof(uint32_t)];
    [c compute:^(id<MTLComputeCommandEncoder> e) {
        [e setComputePipelineState:pipeline];
        [e setBuffer:out offset:0 atIndex:0];
        if (bindExtra) [e setBuffer:extra offset:0 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }];
    if (extraOut) *extraOut = [NSData dataWithBytes:extra.contents length:extra.length];
    return [NSData dataWithBytes:out.contents length:out.length];
}

WMREF_CASE(functionConstants, "function.constants", "function.constants",
           "Function constant reflection, specialization by index and by name, optional constants")
{
    id<MTLFunction> generic = [c function:@"pip_constants"];
    [c value:@"generic.constants" strings:WMDescribeConstants(generic.functionConstantsDictionary) rule:WMExact()];

    NSError *error = nil;
    id<MTLFunction> specialized = [c.library newFunctionWithName:@"pip_constants"
                                                  constantValues:WMBaseConstants(YES) error:&error];
    [c error:@"flag-on.error" value:error rule:WMExact()];
    if (!specialized) [c fail:@"specialization failed: %@", error];
    [c value:@"flag-on.name" string:specialized.name rule:WMExact()];
    [c value:@"flag-on.constants" strings:WMDescribeConstants(specialized.functionConstantsDictionary)
        rule:WMExact()];
    NSData *extra = nil;
    NSData *out = WMRunConstants(c, specialized, YES, &extra);
    [c data:@"flag-on.results" elem:WMElemU32 bytes:out.bytes length:out.length rule:WMExact()];
    [c data:@"flag-on.extra" elem:WMElemU32 bytes:extra.bytes length:extra.length rule:WMExact()];

    // The flag off removes the buffer(1) argument; the optional constant is
    // supplied.
    MTLFunctionConstantValues *values = WMBaseConstants(NO);
    const uint32_t optional = 42;
    [values setConstantValue:&optional type:MTLDataTypeUInt atIndex:3];
    error = nil;
    specialized = [c.library newFunctionWithName:@"pip_constants" constantValues:values error:&error];
    if (!specialized) [c fail:@"specialization failed: %@", error];
    out = WMRunConstants(c, specialized, NO, NULL);
    [c data:@"flag-off.results" elem:WMElemU32 bytes:out.bytes length:out.length rule:WMExact()];

    // setConstantValues:type:withRange: over consecutive indices.
    values = WMBaseConstants(NO);
    const uint32_t pair[2] = {0xabc, 0x5};
    [values setConstantValues:pair type:MTLDataTypeUInt withRange:NSMakeRange(3, 1)];
    [values setConstantValues:pair type:MTLDataTypeUInt withRange:NSMakeRange(0, 1)];
    specialized = [c.library newFunctionWithName:@"pip_constants" constantValues:values error:&error];
    if (!specialized) [c fail:@"range specialization failed: %@", error];
    out = WMRunConstants(c, specialized, NO, NULL);
    [c data:@"range.results" elem:WMElemU32 bytes:out.bytes length:out.length rule:WMExact()];

    // Constant 0 is required (used without is_function_constant_defined).
    values = [MTLFunctionConstantValues new];
    const bool flag = false;
    [values setConstantValue:&flag type:MTLDataTypeBool atIndex:1];
    error = nil;
    specialized = [c.library newFunctionWithName:@"pip_constants" constantValues:values error:&error];
    [c value:@"missing-required.nil" boolean:specialized == nil rule:WMExact()];
    [c error:@"missing-required.error" value:error rule:WMExact()];
    if (specialized) {
        error = nil;
        id<MTLComputePipelineState> state = [c.device newComputePipelineStateWithFunction:specialized error:&error];
        [c value:@"missing-required.pipeline-nil" boolean:state == nil rule:WMExact()];
        [c error:@"missing-required.pipeline-error" value:WMRefStableError(error) rule:WMRefDriverErrorRule()];
    }
}

WMREF_CASE(functionDescriptor, "function.descriptor", "function.constants",
           "newFunctionWithDescriptor:error: with a specialized name and constant values")
{
    MTLFunctionDescriptor *descriptor = [MTLFunctionDescriptor functionDescriptor];
    descriptor.name = @"pip_constants";
    descriptor.specializedName = @"pip_constants_specialized";
    descriptor.constantValues = WMBaseConstants(YES);
    NSError *error = nil;
    id<MTLFunction> function = [c.library newFunctionWithDescriptor:descriptor error:&error];
    [c error:@"error" value:error rule:WMExact()];
    if (!function) [c fail:@"descriptor function: %@", error];
    [c value:@"name" string:function.name rule:WMExact()];
    [c value:@"options" u64:function.options rule:WMExact()];
    NSData *extra = nil;
    NSData *out = WMRunConstants(c, function, YES, &extra);
    [c data:@"results" elem:WMElemU32 bytes:out.bytes length:out.length rule:WMExact()];
    [c data:@"extra" elem:WMElemU32 bytes:extra.bytes length:extra.length rule:WMExact()];
}

static NSArray<NSString *> *WMDescribeAttributes(NSArray<MTLAttribute *> *attributes) {
    NSMutableArray<NSString *> *entries = [NSMutableArray array];
    for (MTLAttribute *attribute in attributes)
        [entries addObject:[NSString stringWithFormat:@"%lu %@ type=%lu active=%d patch=%d control-point=%d",
                            (unsigned long)attribute.attributeIndex, attribute.name,
                            (unsigned long)attribute.attributeType, attribute.active, attribute.patchData,
                            attribute.patchControlPointData]];
    return WMRefSortedNames(entries);
}

WMREF_CASE(functionReflection, "function.reflection", "function.reflection",
           "MTLFunction type, vertex and stage-input attributes and patch information")
{
    struct { const char *name; } functions[] = {
        {"pip_reflect_vertex"}, {"pip_reflect_fragment"}, {"pip_reflect_compute"}, {"pip_post_tess"},
        {"pip_stage_kernel"}, {"pip_void_vertex"},
    };
    for (size_t i = 0; i < sizeof functions / sizeof functions[0]; i++) {
        id<MTLFunction> function = [c function:@(functions[i].name)];
        NSString *base = @(functions[i].name);
        [c value:[base stringByAppendingString:@".type"] i64:function.functionType rule:WMExact()];
        [c value:[base stringByAppendingString:@".name"] string:function.name rule:WMExact()];
        [c value:[base stringByAppendingString:@".same-device"] boolean:function.device == c.device rule:WMExact()];
        [c value:[base stringByAppendingString:@".patch-type"] i64:function.patchType rule:WMExact()];
        [c value:[base stringByAppendingString:@".patch-control-points"] i64:function.patchControlPointCount
            rule:WMExact()];
        [c value:[base stringByAppendingString:@".options"] u64:function.options rule:WMExact()];
        [c value:[base stringByAppendingString:@".vertex-attributes"]
         strings:WMDescribeAttributes(function.vertexAttributes ?: @[]) rule:WMExact()];
        [c value:[base stringByAppendingString:@".stage-input-attributes"]
         strings:WMDescribeAttributes(function.stageInputAttributes ?: @[]) rule:WMExact()];
    }
}
