// Metal reference harness: interface for writing cases.
//
// A case is a function registered with WMREF_CASE. It receives a WMCase
// object that owns the device under test and records outputs. Every output
// carries a comparison rule (FORMAT.md, "Rules"); there is no default rule.
//
//   WMREF_CASE(copyBuffer, "blit.copy-buffer", "blit.buffer-to-buffer",
//              "Copy a range between two buffers")
//   {
//       ...
//       [c data:@"destination" elem:WMElemU8 bytes:p length:n rule:WMExact()];
//   }

#import "WMRef.h"

NS_ASSUME_NONNULL_BEGIN

typedef NSDictionary<NSString *, id> WMRule;

// The value must equal the reference value.
WMRule *WMExact(void);
// The value is a property of the device (limits, capability answers). It is
// compared only between runs on the same device profile.
WMRule *WMDevice(void);
// The value is recorded for reference and not compared. `reason` states why
// (for example: implementation-defined in Metal, OS-specific text).
WMRule *WMNone(NSString *reason);
// The value must be one of `values` (JSON scalars). `reason` cites the
// definition that allows each of them.
WMRule *WMOneOf(NSArray *values, NSString *reason);

typedef NS_ENUM(NSInteger, WMElem) {
    WMElemU8, WMElemU16, WMElemU32, WMElemU64,
    WMElemI8, WMElemI16, WMElemI32, WMElemI64,
    WMElemF16, WMElemF32, WMElemF64,
};
NSString *WMElemName(WMElem elem);
size_t WMElemSize(WMElem elem);

@interface WMCase : NSObject

@property (nonatomic, readonly) id<MTLDevice> device;
// Harness shader library (shaders/*.metal), loaded on first use.
@property (nonatomic, readonly) id<MTLLibrary> library;
// A command queue created on first use.
@property (nonatomic, readonly) id<MTLCommandQueue> queue;

// Records a requirement. A false value ends the case with status
// "unsupported"; nothing after it runs.
- (void)require:(NSString *)name value:(BOOL)value;
// Records a progress marker, so a run that terminates shows the last call
// that was reached.
- (void)step:(NSString *)name;
// Ends the case with status "harness-error". Used when the harness itself
// cannot continue (a pipeline it needs failed to build, for example).
- (void)fail:(NSString *)format, ... NS_FORMAT_FUNCTION(1, 2) __attribute__((noreturn));

// Scalar values.
- (void)value:(NSString *)name u64:(uint64_t)v rule:(WMRule *)rule;
- (void)value:(NSString *)name i64:(int64_t)v rule:(WMRule *)rule;
- (void)value:(NSString *)name boolean:(BOOL)v rule:(WMRule *)rule;
- (void)value:(NSString *)name string:(nullable NSString *)v rule:(WMRule *)rule;
- (void)value:(NSString *)name strings:(NSArray<NSString *> *)v rule:(WMRule *)rule;
- (void)value:(NSString *)name f32:(float)v rule:(WMRule *)rule;
- (void)value:(NSString *)name f64:(double)v rule:(WMRule *)rule;

// Byte data interpreted as `elem` little-endian elements. `layout`
// describes texture data (FORMAT.md, "Layouts") and may be nil.
- (void)data:(NSString *)name elem:(WMElem)elem bytes:(const void *)bytes
      length:(size_t)length rule:(WMRule *)rule;
- (void)data:(NSString *)name elem:(WMElem)elem bytes:(const void *)bytes
      length:(size_t)length layout:(nullable NSDictionary *)layout rule:(WMRule *)rule;

// Data compared against ideal values under an error model (FORMAT.md,
// "Bounded comparisons"). `ideal` holds `count` doubles. `model` is one of
// "ulp", "abs", "srgb8-decode" (floating-point elements) or "srgb8-encode"
// (u8 codes); `bound` is the decimal bound for the model. `reference` names
// the clause that states the bound.
- (void)bounded:(NSString *)name elem:(WMElem)elem bytes:(const void *)bytes
          count:(size_t)count ideal:(const double *)ideal model:(NSString *)model
          bound:(NSString *)bound reference:(NSString *)reference;

// An NSError result: nil-ness, domain and code are compared by `rule`; the
// localized description is always recorded as reference text only.
- (void)error:(NSString *)name value:(nullable NSError *)error rule:(WMRule *)rule;

// Runs `block` and records whether it raised an Objective-C exception and,
// if so, the exception name (compared by `rule`) and reason (reference text
// only). Returns YES when the block returned normally.
- (BOOL)call:(NSString *)name rule:(WMRule *)rule block:(void (^)(void))block;

// Helpers that end the case with "harness-error" when Metal returns nil.
- (id<MTLFunction>)function:(NSString *)name;
- (id<MTLFunction>)function:(NSString *)name constants:(nullable MTLFunctionConstantValues *)constants;
- (id<MTLComputePipelineState>)computePipeline:(NSString *)name;
- (id<MTLComputePipelineState>)computePipeline:(NSString *)name
                                     constants:(nullable MTLFunctionConstantValues *)constants;
- (id<MTLRenderPipelineState>)renderPipeline:(MTLRenderPipelineDescriptor *)descriptor;
- (id<MTLDepthStencilState>)depthStencilState:(MTLDepthStencilDescriptor *)descriptor;
- (id<MTLSamplerState>)sampler:(MTLSamplerDescriptor *)descriptor;
- (id<MTLTexture>)texture:(MTLTextureDescriptor *)descriptor;
// Shared-storage buffer whose contents are cleared by the CPU.
- (id<MTLBuffer>)buffer:(NSUInteger)length;
- (id<MTLBuffer>)bufferWithBytes:(const void *)bytes length:(NSUInteger)length;
- (id<MTLCommandBuffer>)commandBuffer;
// Commits and waits; ends the case with "harness-error" unless the
// command buffer completes without error.
- (void)commitAndWait:(id<MTLCommandBuffer>)commandBuffer;

// One command buffer with one encoder, committed and waited for.
- (void)compute:(void (^)(id<MTLComputeCommandEncoder> encoder))body;
- (void)blit:(void (^)(id<MTLBlitCommandEncoder> encoder))body;
- (void)render:(MTLRenderPassDescriptor *)pass body:(void (^)(id<MTLRenderCommandEncoder> encoder))body;

// Copies one level and slice (all depth images for 3D) of a texture into a
// buffer with the given row and image pitch, through a blit, and returns
// the bytes. Works for every storage mode.
- (NSData *)readTexture:(id<MTLTexture>)texture level:(NSUInteger)level slice:(NSUInteger)slice
            bytesPerRow:(NSUInteger)bytesPerRow;
// As above with an explicit row count per image (block rows for
// compressed formats).
- (NSData *)readTexture:(id<MTLTexture>)texture level:(NSUInteger)level slice:(NSUInteger)slice
            bytesPerRow:(NSUInteger)bytesPerRow rows:(NSUInteger)rows;

@end

// Registration.
typedef void (*WMRefCaseFunction)(WMCase *c);
void WMRefRegisterCase(const char *identifier, const char *area, const char *title,
                       WMRefCaseFunction function);

#define WMREF_CASE(function, identifier, area, title)                                   \
    static void function(WMCase *c);                                                     \
    __attribute__((constructor)) static void function##Register(void) {                  \
        WMRefRegisterCase(identifier, area, title, function);                            \
    }                                                                                    \
    static void function(WMCase *c)

// Deterministic test data: xorshift32 bytes from `seed` (FORMAT.md,
// "Generated inputs").
void WMPattern(void *destination, size_t length, uint32_t seed);

// Name of a pixel format as spelled in MTLPixelFormat.h without the
// "MTLPixelFormat" prefix, or nil for a value the table does not contain.
NSString *_Nullable WMPixelFormatName(MTLPixelFormat format);

// IEEE binary16 helpers for building inputs and ideal values.
double WMHalfToDouble(uint16_t bits);
uint16_t WMHalfFromFloat(float value); // round to nearest even

NS_ASSUME_NONNULL_END
