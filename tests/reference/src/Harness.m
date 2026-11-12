// Metal reference harness: case registry, record writer and helpers.

#import "WMRefCase.h"
#import "WMRefInternal.h"
#import "WMRefBuildInfo.h"

#include <sys/sysctl.h>
#include <TargetConditionals.h>

NSString *const WMRefFormatVersion = @"webmetal-reference/1";

static NSString *const WMRefUnsupportedException = @"WMRefUnsupported";
static NSString *const WMRefFailureException = @"WMRefFailure";

#pragma mark - Registry

typedef struct {
    WMRefCaseInfo info;
    WMRefCaseFunction function;
} WMRefEntry;

static WMRefEntry *gEntries;
static NSUInteger gEntryCount;
static NSUInteger gEntryCapacity;
static BOOL gEntriesSorted;

void WMRefRegisterCase(const char *identifier, const char *area, const char *title,
                       WMRefCaseFunction function) {
    if (gEntryCount == gEntryCapacity) {
        gEntryCapacity = gEntryCapacity ? gEntryCapacity * 2 : 64;
        gEntries = realloc(gEntries, gEntryCapacity * sizeof(WMRefEntry));
        if (!gEntries) abort();
    }
    gEntries[gEntryCount++] = (WMRefEntry){{identifier, area, title}, function};
    gEntriesSorted = NO;
}

static int WMRefEntryCompare(const void *a, const void *b) {
    return strcmp(((const WMRefEntry *)a)->info.identifier, ((const WMRefEntry *)b)->info.identifier);
}

static void WMRefSortEntries(void) {
    if (gEntriesSorted) return;
    qsort(gEntries, gEntryCount, sizeof(WMRefEntry), WMRefEntryCompare);
    for (NSUInteger i = 1; i < gEntryCount; i++) {
        if (strcmp(gEntries[i - 1].info.identifier, gEntries[i].info.identifier) == 0) {
            fprintf(stderr, "wmref: case \"%s\" is registered twice\n", gEntries[i].info.identifier);
            abort();
        }
    }
    gEntriesSorted = YES;
}

NSUInteger WMRefCaseCount(void) {
    WMRefSortEntries();
    return gEntryCount;
}

WMRefCaseInfo WMRefCaseAtIndex(NSUInteger index) {
    WMRefSortEntries();
    if (index >= gEntryCount) abort();
    return gEntries[index].info;
}

static const WMRefEntry *WMRefLookup(const char *identifier) {
    WMRefSortEntries();
    WMRefEntry key = {{identifier, NULL, NULL}, NULL};
    return bsearch(&key, gEntries, gEntryCount, sizeof(WMRefEntry), WMRefEntryCompare);
}

BOOL WMRefFindCase(const char *identifier, WMRefCaseInfo *info) {
    const WMRefEntry *entry = WMRefLookup(identifier);
    if (entry && info) *info = entry->info;
    return entry != NULL;
}

#pragma mark - Canonical JSON

static void WMRefAppendJSONString(NSMutableString *out, NSString *string) {
    [out appendString:@"\""];
    NSUInteger length = string.length;
    for (NSUInteger i = 0; i < length; i++) {
        unichar ch = [string characterAtIndex:i];
        switch (ch) {
        case '"': [out appendString:@"\\\""]; break;
        case '\\': [out appendString:@"\\\\"]; break;
        case '\n': [out appendString:@"\\n"]; break;
        case '\r': [out appendString:@"\\r"]; break;
        case '\t': [out appendString:@"\\t"]; break;
        default:
            if (ch < 0x20 || ch == 0x7f) [out appendFormat:@"\\u%04x", ch];
            else [out appendFormat:@"%C", ch];
        }
    }
    [out appendString:@"\""];
}

static void WMRefAppendJSON(NSMutableString *out, id value) {
    if (value == nil || value == (id)kCFNull) {
        [out appendString:@"null"];
    } else if ([value isKindOfClass:[NSString class]]) {
        WMRefAppendJSONString(out, value);
    } else if ([value isKindOfClass:[NSNumber class]]) {
        NSNumber *number = value;
        if (CFGetTypeID((__bridge CFTypeRef)number) == CFBooleanGetTypeID()) {
            [out appendString:number.boolValue ? @"true" : @"false"];
            return;
        }
        const char *type = number.objCType;
        switch (type[0]) {
        case 'c': case 's': case 'i': case 'l': case 'q':
            [out appendFormat:@"%lld", number.longLongValue];
            break;
        case 'C': case 'S': case 'I': case 'L': case 'Q':
            [out appendFormat:@"%llu", number.unsignedLongLongValue];
            break;
        default:
            // Floating-point values are written as bit patterns by the
            // record writer; a float reaching here is a harness bug.
            fprintf(stderr, "wmref: non-integer number in JSON output\n");
            abort();
        }
    } else if ([value isKindOfClass:[NSArray class]]) {
        [out appendString:@"["];
        BOOL first = YES;
        for (id element in (NSArray *)value) {
            if (!first) [out appendString:@","];
            first = NO;
            WMRefAppendJSON(out, element);
        }
        [out appendString:@"]"];
    } else if ([value isKindOfClass:[NSDictionary class]]) {
        NSDictionary *dictionary = value;
        NSArray *keys = [dictionary.allKeys sortedArrayUsingSelector:@selector(compare:)];
        [out appendString:@"{"];
        BOOL first = YES;
        for (NSString *key in keys) {
            if (!first) [out appendString:@","];
            first = NO;
            WMRefAppendJSONString(out, key);
            [out appendString:@":"];
            WMRefAppendJSON(out, dictionary[key]);
        }
        [out appendString:@"}"];
    } else {
        fprintf(stderr, "wmref: unsupported JSON value of class %s\n",
                NSStringFromClass([value class]).UTF8String);
        abort();
    }
}

NSString *WMRefJSON(id value) {
    NSMutableString *out = [NSMutableString string];
    WMRefAppendJSON(out, value);
    return out;
}

static NSString *WMHex(const void *bytes, size_t length) {
    static const char digits[] = "0123456789abcdef";
    const uint8_t *p = bytes;
    char *text = malloc(length * 2 + 1);
    if (!text) abort();
    for (size_t i = 0; i < length; i++) {
        text[2 * i] = digits[p[i] >> 4];
        text[2 * i + 1] = digits[p[i] & 15];
    }
    text[length * 2] = 0;
    NSString *string = [[NSString alloc] initWithBytesNoCopy:text length:length * 2
                                                    encoding:NSASCIIStringEncoding freeWhenDone:YES];
    return string;
}

#pragma mark - Rules

WMRule *WMExact(void) { return @{@"compare": @"exact"}; }
WMRule *WMDevice(void) { return @{@"compare": @"device"}; }
WMRule *WMNone(NSString *reason) { return @{@"compare": @"none", @"reason": reason}; }
WMRule *WMOneOf(NSArray *values, NSString *reason) {
    return @{@"compare": @"property", @"property": @"one-of", @"values": values, @"reason": reason};
}

NSString *WMElemName(WMElem elem) {
    static NSString *const names[] = {
        @"u8", @"u16", @"u32", @"u64", @"i8", @"i16", @"i32", @"i64", @"f16", @"f32", @"f64",
    };
    if ((NSUInteger)elem >= sizeof(names) / sizeof(names[0])) abort();
    return names[elem];
}

size_t WMElemSize(WMElem elem) {
    static const size_t sizes[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};
    if ((NSUInteger)elem >= sizeof(sizes) / sizeof(sizes[0])) abort();
    return sizes[elem];
}

#pragma mark - Case object

@interface WMCase ()
- (instancetype)initWithDevice:(id<MTLDevice>)device libraryURL:(NSURL *)libraryURL
                          sink:(WMRefRecordSink)sink;
- (void)emit:(NSDictionary *)record;
@end

@implementation WMCase {
    WMRefRecordSink _sink;
    id<MTLLibrary> _library;
    id<MTLCommandQueue> _queue;
    NSMutableSet<NSString *> *_outputNames;
    NSMutableSet<NSString *> *_requireNames;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device libraryURL:(NSURL *)libraryURL
                          sink:(WMRefRecordSink)sink {
    if ((self = [super init])) {
        _device = device;
        _libraryURL = libraryURL;
        _sink = sink;
        _outputNames = [NSMutableSet set];
        _requireNames = [NSMutableSet set];
    }
    return self;
}

- (id<MTLLibrary>)library {
    @synchronized(self) {
        if (!_library) {
            NSError *error = nil;
            _library = [_device newLibraryWithURL:_libraryURL error:&error];
            if (!_library) [self fail:@"cannot load %@: %@", _libraryURL.path, error];
        }
        return _library;
    }
}

- (id<MTLCommandQueue>)queue {
    @synchronized(self) {
        if (!_queue) {
            _queue = [_device newCommandQueue];
            if (!_queue) [self fail:@"newCommandQueue returned nil"];
        }
        return _queue;
    }
}

- (void)emit:(NSDictionary *)record {
    NSString *line = WMRefJSON(record);
    @synchronized(self) {
        _sink(line);
    }
}

- (void)claimOutput:(NSString *)name {
    @synchronized(self) {
        if ([_outputNames containsObject:name]) [self fail:@"output \"%@\" recorded twice", name];
        [_outputNames addObject:name];
    }
}

- (void)require:(NSString *)name value:(BOOL)value {
    @synchronized(self) {
        if ([_requireNames containsObject:name]) [self fail:@"requirement \"%@\" recorded twice", name];
        [_requireNames addObject:name];
    }
    [self emit:@{@"kind": @"require", @"name": name, @"value": @(value)}];
    if (!value) @throw [NSException exceptionWithName:WMRefUnsupportedException reason:name userInfo:nil];
}

- (void)step:(NSString *)name {
    [self emit:@{@"kind": @"step", @"name": name}];
}

- (void)fail:(NSString *)format, ... {
    va_list arguments;
    va_start(arguments, format);
    NSString *message = [[NSString alloc] initWithFormat:format arguments:arguments];
    va_end(arguments);
    @throw [NSException exceptionWithName:WMRefFailureException reason:message userInfo:nil];
}

- (void)value:(NSString *)name type:(NSString *)type value:(id)value display:(NSString *)display
         rule:(WMRule *)rule {
    [self claimOutput:name];
    NSMutableDictionary *record = [@{@"kind": @"value", @"name": name, @"type": type,
                                     @"value": value ?: [NSNull null], @"rule": rule} mutableCopy];
    if (display) record[@"display"] = display;
    [self emit:record];
}

- (void)value:(NSString *)name u64:(uint64_t)v rule:(WMRule *)rule {
    [self value:name type:@"u64" value:@(v) display:nil rule:rule];
}

- (void)value:(NSString *)name i64:(int64_t)v rule:(WMRule *)rule {
    [self value:name type:@"i64" value:@(v) display:nil rule:rule];
}

- (void)value:(NSString *)name boolean:(BOOL)v rule:(WMRule *)rule {
    [self value:name type:@"bool" value:@(v) display:nil rule:rule];
}

- (void)value:(NSString *)name string:(NSString *)v rule:(WMRule *)rule {
    [self value:name type:@"string" value:v display:nil rule:rule];
}

- (void)value:(NSString *)name strings:(NSArray<NSString *> *)v rule:(WMRule *)rule {
    [self value:name type:@"strings" value:v display:nil rule:rule];
}

- (void)value:(NSString *)name f32:(float)v rule:(WMRule *)rule {
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    [self value:name type:@"f32" value:[NSString stringWithFormat:@"0x%08x", bits]
        display:[NSString stringWithFormat:@"%.9g", v] rule:rule];
}

- (void)value:(NSString *)name f64:(double)v rule:(WMRule *)rule {
    uint64_t bits;
    memcpy(&bits, &v, sizeof bits);
    [self value:name type:@"f64" value:[NSString stringWithFormat:@"0x%016llx", bits]
        display:[NSString stringWithFormat:@"%.17g", v] rule:rule];
}

- (void)data:(NSString *)name elem:(WMElem)elem bytes:(const void *)bytes length:(size_t)length
        rule:(WMRule *)rule {
    [self data:name elem:elem bytes:bytes length:length layout:nil rule:rule];
}

- (void)data:(NSString *)name elem:(WMElem)elem bytes:(const void *)bytes length:(size_t)length
      layout:(NSDictionary *)layout rule:(WMRule *)rule {
    size_t size = WMElemSize(elem);
    if (length % size) [self fail:@"data \"%@\": %zu bytes is not a whole number of %@", name, length, WMElemName(elem)];
    [self claimOutput:name];
    NSMutableDictionary *record = [@{@"kind": @"data", @"name": name, @"elem": WMElemName(elem),
                                     @"count": @(length / size), @"hex": WMHex(bytes, length),
                                     @"rule": rule} mutableCopy];
    if (layout) record[@"layout"] = layout;
    [self emit:record];
}

- (void)bounded:(NSString *)name elem:(WMElem)elem bytes:(const void *)bytes count:(size_t)count
          ideal:(const double *)ideal model:(NSString *)model bound:(NSString *)bound
      reference:(NSString *)reference {
    if (![@[@"ulp", @"abs", @"srgb8-decode", @"srgb8-encode"] containsObject:model])
        [self fail:@"bounded \"%@\": unknown model %@", name, model];
    BOOL codes = [model isEqualToString:@"srgb8-encode"];
    if (codes ? elem != WMElemU8 : (elem != WMElemF16 && elem != WMElemF32 && elem != WMElemF64))
        [self fail:@"bounded \"%@\": element type %@ does not fit model %@", name, WMElemName(elem), model];
    WMRule *rule = @{@"compare": @"bounded", @"model": model, @"bound": bound, @"reference": reference,
                     @"ideal": WMHex(ideal, count * sizeof(double))};
    [self data:name elem:elem bytes:bytes length:count * WMElemSize(elem) layout:nil rule:rule];
}

- (void)error:(NSString *)name value:(NSError *)error rule:(WMRule *)rule {
    [self claimOutput:name];
    NSMutableDictionary *record = [@{@"kind": @"error", @"name": name, @"nil": @(error == nil),
                                     @"rule": rule} mutableCopy];
    if (error) {
        record[@"domain"] = error.domain;
        record[@"code"] = @(error.code);
        record[@"description"] = error.localizedDescription ?: @"";
    }
    [self emit:record];
}

- (BOOL)call:(NSString *)name rule:(WMRule *)rule block:(void (^)(void))block {
    NSException *raised = nil;
    @try {
        block();
    } @catch (NSException *exception) {
        if ([exception.name isEqualToString:WMRefUnsupportedException] ||
            [exception.name isEqualToString:WMRefFailureException])
            @throw;
        raised = exception;
    }
    [self claimOutput:name];
    NSMutableDictionary *record = [@{@"kind": @"call", @"name": name, @"raised": @(raised != nil),
                                     @"rule": rule} mutableCopy];
    if (raised) {
        record[@"exception"] = raised.name;
        record[@"reason"] = raised.reason ?: @"";
    }
    [self emit:record];
    return raised == nil;
}

#pragma mark Helpers

- (id<MTLFunction>)function:(NSString *)name {
    id<MTLFunction> function = [self.library newFunctionWithName:name];
    if (!function) [self fail:@"function %@ not found in the harness library", name];
    return function;
}

- (id<MTLFunction>)function:(NSString *)name constants:(MTLFunctionConstantValues *)constants {
    if (!constants) return [self function:name];
    NSError *error = nil;
    id<MTLFunction> function = [self.library newFunctionWithName:name constantValues:constants error:&error];
    if (!function) [self fail:@"function %@ with constants: %@", name, error];
    return function;
}

- (id<MTLComputePipelineState>)computePipeline:(NSString *)name {
    return [self computePipeline:name constants:nil];
}

- (id<MTLComputePipelineState>)computePipeline:(NSString *)name constants:(MTLFunctionConstantValues *)constants {
    NSError *error = nil;
    id<MTLComputePipelineState> state =
        [_device newComputePipelineStateWithFunction:[self function:name constants:constants] error:&error];
    if (!state) [self fail:@"compute pipeline %@: %@", name, error];
    return state;
}

- (id<MTLRenderPipelineState>)renderPipeline:(MTLRenderPipelineDescriptor *)descriptor {
    NSError *error = nil;
    id<MTLRenderPipelineState> state = [_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (!state) [self fail:@"render pipeline: %@", error];
    return state;
}

- (id<MTLDepthStencilState>)depthStencilState:(MTLDepthStencilDescriptor *)descriptor {
    id<MTLDepthStencilState> state = [_device newDepthStencilStateWithDescriptor:descriptor];
    if (!state) [self fail:@"newDepthStencilStateWithDescriptor returned nil"];
    return state;
}

- (id<MTLSamplerState>)sampler:(MTLSamplerDescriptor *)descriptor {
    id<MTLSamplerState> state = [_device newSamplerStateWithDescriptor:descriptor];
    if (!state) [self fail:@"newSamplerStateWithDescriptor returned nil"];
    return state;
}

- (id<MTLTexture>)texture:(MTLTextureDescriptor *)descriptor {
    id<MTLTexture> texture = [_device newTextureWithDescriptor:descriptor];
    if (!texture) [self fail:@"newTextureWithDescriptor returned nil for %@", WMPixelFormatName(descriptor.pixelFormat)];
    return texture;
}

- (id<MTLBuffer>)buffer:(NSUInteger)length {
    id<MTLBuffer> buffer = [_device newBufferWithLength:length options:MTLResourceStorageModeShared];
    if (!buffer) [self fail:@"newBufferWithLength:%lu returned nil", (unsigned long)length];
    memset(buffer.contents, 0, length);
    return buffer;
}

- (id<MTLBuffer>)bufferWithBytes:(const void *)bytes length:(NSUInteger)length {
    id<MTLBuffer> buffer = [_device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
    if (!buffer) [self fail:@"newBufferWithBytes:length:%lu returned nil", (unsigned long)length];
    return buffer;
}

- (id<MTLCommandBuffer>)commandBuffer {
    id<MTLCommandBuffer> commandBuffer = [self.queue commandBuffer];
    if (!commandBuffer) [self fail:@"commandBuffer returned nil"];
    return commandBuffer;
}

- (void)commitAndWait:(id<MTLCommandBuffer>)commandBuffer {
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted)
        [self fail:@"command buffer ended with status %ld: %@", (long)commandBuffer.status, commandBuffer.error];
}

- (void)compute:(void (^)(id<MTLComputeCommandEncoder>))body {
    id<MTLCommandBuffer> commandBuffer = [self commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    body(encoder);
    [encoder endEncoding];
    [self commitAndWait:commandBuffer];
}

- (void)blit:(void (^)(id<MTLBlitCommandEncoder>))body {
    id<MTLCommandBuffer> commandBuffer = [self commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    body(encoder);
    [encoder endEncoding];
    [self commitAndWait:commandBuffer];
}

- (void)render:(MTLRenderPassDescriptor *)pass body:(void (^)(id<MTLRenderCommandEncoder>))body {
    id<MTLCommandBuffer> commandBuffer = [self commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    if (!encoder) [self fail:@"renderCommandEncoderWithDescriptor returned nil"];
    body(encoder);
    [encoder endEncoding];
    [self commitAndWait:commandBuffer];
}

- (NSData *)readTexture:(id<MTLTexture>)texture level:(NSUInteger)level slice:(NSUInteger)slice
            bytesPerRow:(NSUInteger)bytesPerRow {
    NSUInteger rows = MAX((NSUInteger)1, texture.height >> level);
    return [self readTexture:texture level:level slice:slice bytesPerRow:bytesPerRow rows:rows];
}

- (NSData *)readTexture:(id<MTLTexture>)texture level:(NSUInteger)level slice:(NSUInteger)slice
            bytesPerRow:(NSUInteger)bytesPerRow rows:(NSUInteger)rows {
    NSUInteger width = MAX((NSUInteger)1, texture.width >> level);
    NSUInteger height = MAX((NSUInteger)1, texture.height >> level);
    NSUInteger depth = texture.textureType == MTLTextureType3D ? MAX((NSUInteger)1, texture.depth >> level) : 1;
    NSUInteger bytesPerImage = bytesPerRow * rows;
    id<MTLBuffer> buffer = [self buffer:bytesPerImage * depth];
    [self blit:^(id<MTLBlitCommandEncoder> encoder) {
        [encoder copyFromTexture:texture sourceSlice:slice sourceLevel:level
                    sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, depth)
                        toBuffer:buffer destinationOffset:0 destinationBytesPerRow:bytesPerRow
         destinationBytesPerImage:bytesPerImage];
    }];
    return [NSData dataWithBytes:buffer.contents length:buffer.length];
}

@end

#pragma mark - Running

BOOL WMRefDeviceInBaseline(id<MTLDevice> device) {
    return [device supportsFamily:MTLGPUFamilyApple3];
}

NSString *WMRefRunCase(id<MTLDevice> device, NSURL *libraryURL, const char *identifier,
                       WMRefRecordSink sink) {
    const WMRefEntry *entry = WMRefLookup(identifier);
    if (!entry) {
        sink(WMRefJSON(@{@"kind": @"end", @"status": @"harness-error",
                         @"message": [NSString stringWithFormat:@"unknown case %s", identifier]}));
        return @"harness-error";
    }
    if (!WMRefDeviceInBaseline(device)) {
        sink(WMRefJSON(@{@"kind": @"end", @"status": @"harness-error",
                         @"message": [NSString stringWithFormat:@"%@ is not in GPU family Apple3 or later",
                                                                device.name]}));
        return @"harness-error";
    }
    WMCase *c = [[WMCase alloc] initWithDevice:device libraryURL:libraryURL sink:sink];
    NSString *status = @"ok";
    NSString *message = nil;
    @try {
        @autoreleasepool {
            entry->function(c);
        }
    } @catch (NSException *exception) {
        if ([exception.name isEqualToString:WMRefUnsupportedException]) {
            status = @"unsupported";
            message = [NSString stringWithFormat:@"requirement %@ is not met", exception.reason];
        } else if ([exception.name isEqualToString:WMRefFailureException]) {
            status = @"harness-error";
            message = exception.reason;
        } else {
            status = @"harness-error";
            message = [NSString stringWithFormat:@"uncaught %@: %@", exception.name, exception.reason];
        }
    }
    NSMutableDictionary *end = [@{@"kind": @"end", @"status": status} mutableCopy];
    if (message) end[@"message"] = message;
    [c emit:end];
    return status;
}

#pragma mark - Identity

NSDictionary *WMRefBuildInfo(void) {
    NSData *data = [@WMREF_BUILD_INFO_JSON dataUsingEncoding:NSUTF8StringEncoding];
    NSDictionary *info = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
    if (![info isKindOfClass:[NSDictionary class]]) {
        fprintf(stderr, "wmref: generated build description is not a JSON object\n");
        abort();
    }
    return info;
}

NSDictionary *WMRefDeviceIdentity(id<MTLDevice> device) {
    NSMutableDictionary *families = [NSMutableDictionary dictionary];
    for (NSInteger family = 1001; family <= 1011; family++)
        families[[NSString stringWithFormat:@"apple%ld", (long)(family - 1000)]] =
            @([device supportsFamily:(MTLGPUFamily)family]);
    families[@"mac2"] = @([device supportsFamily:(MTLGPUFamily)2002]);
    for (NSInteger family = 3001; family <= 3003; family++)
        families[[NSString stringWithFormat:@"common%ld", (long)(family - 3000)]] =
            @([device supportsFamily:(MTLGPUFamily)family]);
    families[@"metal3"] = @([device supportsFamily:MTLGPUFamilyMetal3]);
    if (@available(macOS 26.0, iOS 26.0, *))
        families[@"metal4"] = @([device supportsFamily:MTLGPUFamilyMetal4]);
    NSMutableDictionary *identity = [@{@"name": device.name, @"families": families} mutableCopy];
    if (@available(macOS 14.0, iOS 17.0, *)) identity[@"architecture"] = device.architecture.name;
    return identity;
}

static NSString *WMRefSysctlString(const char *name) {
    size_t size = 0;
    if (sysctlbyname(name, NULL, &size, NULL, 0) != 0 || size == 0) return nil;
    char *buffer = malloc(size);
    if (!buffer) abort();
    if (sysctlbyname(name, buffer, &size, NULL, 0) != 0) {
        free(buffer);
        return nil;
    }
    NSString *string = [[NSString alloc] initWithBytes:buffer length:strnlen(buffer, size)
                                              encoding:NSUTF8StringEncoding];
    free(buffer);
    return string;
}

NSDictionary *WMRefHostIdentity(void) {
#if TARGET_OS_OSX
    NSString *platform = @"macOS";
    NSString *model = WMRefSysctlString("hw.model");
#elif TARGET_OS_IOS
    NSString *platform = @"iOS";
    NSString *model = WMRefSysctlString("hw.machine");
#else
#error unsupported platform
#endif
    NSBundle *metal = [NSBundle bundleWithIdentifier:@"com.apple.Metal"];
    return @{
        @"platform": platform,
        @"os_version": WMRefSysctlString("kern.osproductversion") ?: @"",
        @"os_build": WMRefSysctlString("kern.osversion") ?: @"",
        @"kernel": WMRefSysctlString("kern.version") ?: @"",
        @"model": model ?: @"",
        @"metal_framework": metal.infoDictionary[@"CFBundleVersion"] ?: @"",
    };
}

NSString *WMRefDeviceKey(NSDictionary *device, NSDictionary *host) {
    NSString *raw = [NSString stringWithFormat:@"%@ %@ %@ %@", device[@"name"], host[@"platform"],
                     host[@"os_version"], host[@"os_build"]];
    NSMutableString *key = [NSMutableString string];
    BOOL dash = NO;
    for (NSUInteger i = 0; i < raw.length; i++) {
        unichar ch = [raw characterAtIndex:i];
        if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '.') {
            if (dash && key.length) [key appendString:@"-"];
            [key appendFormat:@"%C", ch];
            dash = NO;
        } else if (ch >= 'A' && ch <= 'Z') {
            if (dash && key.length) [key appendString:@"-"];
            [key appendFormat:@"%C", (unichar)(ch - 'A' + 'a')];
            dash = NO;
        } else {
            dash = YES;
        }
    }
    return key;
}

#pragma mark - Utilities

void WMPattern(void *destination, size_t length, uint32_t seed) {
    uint32_t state = seed ? seed : 0x9e3779b9u;
    uint8_t *p = destination;
    for (size_t i = 0; i < length; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        p[i] = (uint8_t)state;
    }
}

double WMHalfToDouble(uint16_t bits) {
    int sign = bits >> 15;
    int exponent = (bits >> 10) & 0x1f;
    int mantissa = bits & 0x3ff;
    double value;
    if (exponent == 0) value = ldexp(mantissa, -24);
    else if (exponent == 31) value = mantissa ? NAN : INFINITY;
    else value = ldexp(mantissa | 0x400, exponent - 25);
    return sign ? -value : value;
}

uint16_t WMHalfFromFloat(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
    uint32_t exponent = (bits >> 23) & 0xff;
    uint32_t mantissa = bits & 0x7fffff;
    if (exponent == 0xff) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
    int e = (int)exponent - 127 + 15;
    if (e >= 31) return sign | 0x7c00;
    if (e <= 0) {
        if (e < -10) return sign;
        uint32_t m = mantissa | 0x800000;
        int shift = 14 - e;
        uint32_t half = m >> shift;
        uint32_t rest = m & ((1u << shift) - 1);
        uint32_t midpoint = 1u << (shift - 1);
        if (rest > midpoint || (rest == midpoint && (half & 1))) half++;
        return sign | (uint16_t)half;
    }
    uint32_t half = ((uint32_t)e << 10) | (mantissa >> 13);
    uint32_t rest = mantissa & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (half & 1))) half++;
    return sign | (uint16_t)half;
}

NSString *WMPixelFormatName(MTLPixelFormat format) {
    // Names and values from MTLPixelFormat.h (SDK 27.2).
    static const struct {
        NSUInteger value;
        __unsafe_unretained NSString *name;
    } table[] = {
    {1, @"A8Unorm"}, {10, @"R8Unorm"}, {11, @"R8Unorm_sRGB"}, {12, @"R8Snorm"}, {13, @"R8Uint"},
    {14, @"R8Sint"}, {20, @"R16Unorm"}, {22, @"R16Snorm"}, {23, @"R16Uint"}, {24, @"R16Sint"},
    {25, @"R16Float"}, {30, @"RG8Unorm"}, {31, @"RG8Unorm_sRGB"}, {32, @"RG8Snorm"},
    {33, @"RG8Uint"}, {34, @"RG8Sint"}, {40, @"B5G6R5Unorm"}, {41, @"A1BGR5Unorm"},
    {42, @"ABGR4Unorm"}, {43, @"BGR5A1Unorm"}, {45, @"RGB8Unorm"}, {46, @"RGB8Snorm"},
    {47, @"RGB8Uint"}, {48, @"RGB8Sint"}, {53, @"R32Uint"}, {54, @"R32Sint"}, {55, @"R32Float"},
    {60, @"RG16Unorm"}, {62, @"RG16Snorm"}, {63, @"RG16Uint"}, {64, @"RG16Sint"},
    {65, @"RG16Float"}, {70, @"RGBA8Unorm"}, {71, @"RGBA8Unorm_sRGB"}, {72, @"RGBA8Snorm"},
    {73, @"RGBA8Uint"}, {74, @"RGBA8Sint"}, {80, @"BGRA8Unorm"}, {81, @"BGRA8Unorm_sRGB"},
    {90, @"RGB10A2Unorm"}, {91, @"RGB10A2Uint"}, {92, @"RG11B10Float"}, {93, @"RGB9E5Float"},
    {94, @"BGR10A2Unorm"}, {554, @"BGR10_XR"}, {555, @"BGR10_XR_sRGB"}, {95, @"RGB16Unorm"},
    {96, @"RGB16Snorm"}, {97, @"RGB16Uint"}, {98, @"RGB16Sint"}, {99, @"RGB16Float"},
    {103, @"RG32Uint"}, {104, @"RG32Sint"}, {105, @"RG32Float"}, {110, @"RGBA16Unorm"},
    {112, @"RGBA16Snorm"}, {113, @"RGBA16Uint"}, {114, @"RGBA16Sint"}, {115, @"RGBA16Float"},
    {552, @"BGRA10_XR"}, {553, @"BGRA10_XR_sRGB"}, {120, @"RGB32Uint"}, {121, @"RGB32Sint"},
    {122, @"RGB32Float"}, {123, @"RGBA32Uint"}, {124, @"RGBA32Sint"}, {125, @"RGBA32Float"},
    {130, @"BC1_RGBA"}, {131, @"BC1_RGBA_sRGB"}, {132, @"BC2_RGBA"}, {133, @"BC2_RGBA_sRGB"},
    {134, @"BC3_RGBA"}, {135, @"BC3_RGBA_sRGB"}, {140, @"BC4_RUnorm"}, {141, @"BC4_RSnorm"},
    {142, @"BC5_RGUnorm"}, {143, @"BC5_RGSnorm"}, {150, @"BC6H_RGBFloat"}, {151, @"BC6H_RGBUfloat"},
    {152, @"BC7_RGBAUnorm"}, {153, @"BC7_RGBAUnorm_sRGB"}, {160, @"PVRTC_RGB_2BPP"},
    {161, @"PVRTC_RGB_2BPP_sRGB"}, {162, @"PVRTC_RGB_4BPP"}, {163, @"PVRTC_RGB_4BPP_sRGB"},
    {164, @"PVRTC_RGBA_2BPP"}, {165, @"PVRTC_RGBA_2BPP_sRGB"}, {166, @"PVRTC_RGBA_4BPP"},
    {167, @"PVRTC_RGBA_4BPP_sRGB"}, {170, @"EAC_R11Unorm"}, {172, @"EAC_R11Snorm"},
    {174, @"EAC_RG11Unorm"}, {176, @"EAC_RG11Snorm"}, {178, @"EAC_RGBA8"}, {179, @"EAC_RGBA8_sRGB"},
    {180, @"ETC2_RGB8"}, {181, @"ETC2_RGB8_sRGB"}, {182, @"ETC2_RGB8A1"},
    {183, @"ETC2_RGB8A1_sRGB"}, {186, @"ASTC_4x4_sRGB"}, {187, @"ASTC_5x4_sRGB"},
    {188, @"ASTC_5x5_sRGB"}, {189, @"ASTC_6x5_sRGB"}, {190, @"ASTC_6x6_sRGB"},
    {192, @"ASTC_8x5_sRGB"}, {193, @"ASTC_8x6_sRGB"}, {194, @"ASTC_8x8_sRGB"},
    {195, @"ASTC_10x5_sRGB"}, {196, @"ASTC_10x6_sRGB"}, {197, @"ASTC_10x8_sRGB"},
    {198, @"ASTC_10x10_sRGB"}, {199, @"ASTC_12x10_sRGB"}, {200, @"ASTC_12x12_sRGB"},
    {204, @"ASTC_4x4_LDR"}, {205, @"ASTC_5x4_LDR"}, {206, @"ASTC_5x5_LDR"}, {207, @"ASTC_6x5_LDR"},
    {208, @"ASTC_6x6_LDR"}, {210, @"ASTC_8x5_LDR"}, {211, @"ASTC_8x6_LDR"}, {212, @"ASTC_8x8_LDR"},
    {213, @"ASTC_10x5_LDR"}, {214, @"ASTC_10x6_LDR"}, {215, @"ASTC_10x8_LDR"},
    {216, @"ASTC_10x10_LDR"}, {217, @"ASTC_12x10_LDR"}, {218, @"ASTC_12x12_LDR"},
    {222, @"ASTC_4x4_HDR"}, {223, @"ASTC_5x4_HDR"}, {224, @"ASTC_5x5_HDR"}, {225, @"ASTC_6x5_HDR"},
    {226, @"ASTC_6x6_HDR"}, {228, @"ASTC_8x5_HDR"}, {229, @"ASTC_8x6_HDR"}, {230, @"ASTC_8x8_HDR"},
    {231, @"ASTC_10x5_HDR"}, {232, @"ASTC_10x6_HDR"}, {233, @"ASTC_10x8_HDR"},
    {234, @"ASTC_10x10_HDR"}, {235, @"ASTC_12x10_HDR"}, {236, @"ASTC_12x12_HDR"}, {240, @"GBGR422"},
    {241, @"BGRG422"}, {250, @"Depth16Unorm"}, {252, @"Depth32Float"}, {253, @"Stencil8"},
    {255, @"Depth24Unorm_Stencil8"}, {260, @"Depth32Float_Stencil8"}, {261, @"X32_Stencil8"},
    {262, @"X24_Stencil8"},
    };
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++)
        if (table[i].value == (NSUInteger)format) return table[i].name;
    return nil;
}
