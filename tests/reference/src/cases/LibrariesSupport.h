// Helpers shared by the library, pipeline, argument buffer, indirect
// command buffer, drawable and validation cases.

#import "WMRefCase.h"

NS_ASSUME_NONNULL_BEGIN

// Waits for `semaphore` for at most `seconds`; returns NO on timeout.
BOOL WMRefWait(dispatch_semaphore_t semaphore, double seconds);

NSArray<NSString *> *WMRefSortedNames(NSArray<NSString *> *names);

// The same domain and code with a localized description in which object
// addresses ("0x" followed by hex digits) and the temporary directory are
// replaced by fixed text, so the recorded description does not change
// between runs.
NSError *_Nullable WMRefStableError(NSError *_Nullable error);
NSString *WMRefStableText(NSString *text);

// Rule for errors produced by a device's shader compiler or driver
// (pipeline creation, binary archives, dynamic libraries): the domain names
// the driver (for example "AGXMetalG16G_B0"), so domain and code are
// compared only between runs on the same device.
WMRule *WMRefDriverErrorRule(void);

// Runs `function` as a kernel with one buffer of `words` 32-bit words at
// index 0, over `threads` threads in one threadgroup, and returns the
// buffer contents.
NSData *WMRefRunWordsKernel(WMCase *c, id<MTLFunction> function, NSUInteger words, NSUInteger threads);

// Canonical text of a pipeline reflection: one line per binding or
// argument with every reflected field, nested structures expanded.
NSArray<NSString *> *WMRefDescribeBindings(NSArray<id<MTLBinding>> *bindings) API_AVAILABLE(macos(13.0), ios(16.0));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
NSArray<NSString *> *WMRefDescribeArguments(NSArray<MTLArgument *> *arguments);
#pragma clang diagnostic pop

NS_ASSUME_NONNULL_END
