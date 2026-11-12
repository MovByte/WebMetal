// Metal reference harness: entry points for running cases against a device.
//
// A case is one small program that exercises one Metal behavior on the
// device it is given and emits records (values, data, errors, progress
// steps). The records are canonical JSON objects, one per line; the format
// is described in FORMAT.md next to this directory.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

// Version string written into every case file and run manifest.
extern NSString *const WMRefFormatVersion;

typedef struct {
    const char *identifier; // e.g. "blit.copy-buffer"
    const char *area;       // API area tag listed in COVERAGE.md
    const char *title;      // one-line description
} WMRefCaseInfo;

// Registered cases, sorted by identifier.
NSUInteger WMRefCaseCount(void);
WMRefCaseInfo WMRefCaseAtIndex(NSUInteger index);
// Returns NO if no case has this identifier.
BOOL WMRefFindCase(const char *identifier, WMRefCaseInfo *_Nullable info);

// Cases use the features that the Metal Feature Set Tables list for GPU
// family Apple3 and earlier without checking for them, and check every
// feature listed for a later family. A device outside that baseline cannot
// run them: WMRefRunCase refuses it and `wmref run` stops.
BOOL WMRefDeviceInBaseline(id<MTLDevice> device);

// Receives one canonical JSON record (without the trailing newline).
typedef void (^WMRefRecordSink)(NSString *line);

// Runs one case in the calling process on `device`. `libraryURL` is the
// harness shader library built for the device's platform. Records are
// passed to `sink` in program order; the last record has kind "end" and
// carries the status ("ok", "unsupported" or "harness-error"), which is
// also returned. A Metal failure that terminates the process ends the run
// without an "end" record; process isolation is the caller's concern
// (see `wmref run`).
NSString *WMRefRunCase(id<MTLDevice> device, NSURL *libraryURL,
                       const char *identifier, WMRefRecordSink sink);

NS_ASSUME_NONNULL_END
