// Metal reference harness: command-line runner.
//
//   wmref list
//   wmref info  [--device SPEC]
//   wmref case  --case ID [--device SPEC] [--library PATH]
//   wmref run   --out DIR [--device SPEC] [--library PATH] [--variants LIST]
//               [--only PREFIXES] [--timeout SECONDS] [--logs DIR]
//
// `run` executes every case in its own process (`wmref child`), once per
// variant, so that a Metal failure that terminates the process is recorded
// as the case's outcome. README.md describes the device specifications.

#import "WMRef.h"
#import "WMRefInternal.h"

#import <CommonCrypto/CommonDigest.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

static void WMDie(NSString *format, ...) NS_FORMAT_FUNCTION(1, 2) __attribute__((noreturn));
static void WMDie(NSString *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    NSString *message = [[NSString alloc] initWithFormat:format arguments:arguments];
    va_end(arguments);
    fprintf(stderr, "wmref: %s\n", message.UTF8String);
    exit(2);
}

#pragma mark - Options

@interface WMOptions : NSObject
@property NSString *command;
@property NSString *caseIdentifier;
@property NSString *device;
@property NSString *library;
@property NSString *output;
@property NSString *logs;
@property NSArray<NSString *> *variants;
@property NSArray<NSString *> *only;
@property double timeout;
@end

@implementation WMOptions
@end

static WMOptions *WMParseOptions(int argc, char **argv) {
    if (argc < 2) WMDie(@"usage: wmref list|info|case|run [options] (see README.md)");
    WMOptions *options = [WMOptions new];
    options.command = @(argv[1]);
    options.device = @"default";
    options.variants = @[@"release", @"debug-layer"];
    options.only = @[];
    options.timeout = 120;
    for (int i = 2; i < argc; i++) {
        NSString *name = @(argv[i]);
        if (i + 1 >= argc) WMDie(@"option %@ needs a value", name);
        NSString *value = @(argv[++i]);
        if ([name isEqualToString:@"--case"]) options.caseIdentifier = value;
        else if ([name isEqualToString:@"--device"]) options.device = value;
        else if ([name isEqualToString:@"--library"]) options.library = value;
        else if ([name isEqualToString:@"--out"]) options.output = value;
        else if ([name isEqualToString:@"--logs"]) options.logs = value;
        else if ([name isEqualToString:@"--variants"]) options.variants = [value componentsSeparatedByString:@","];
        else if ([name isEqualToString:@"--only"]) options.only = [value componentsSeparatedByString:@","];
        else if ([name isEqualToString:@"--timeout"]) options.timeout = value.doubleValue;
        else WMDie(@"unknown option %@", name);
    }
    for (NSString *variant in options.variants)
        if (![variant isEqualToString:@"release"] && ![variant isEqualToString:@"debug-layer"])
            WMDie(@"unknown variant %@ (release, debug-layer)", variant);
    if (options.timeout <= 0) WMDie(@"--timeout must be positive");
    return options;
}

static NSString *WMExecutablePath(void) {
    uint32_t size = 0;
    _NSGetExecutablePath(NULL, &size);
    char *path = malloc(size);
    if (!path || _NSGetExecutablePath(path, &size) != 0) WMDie(@"cannot determine the executable path");
    char *resolved = realpath(path, NULL);
    free(path);
    if (!resolved) WMDie(@"cannot resolve the executable path");
    NSString *result = @(resolved);
    free(resolved);
    return result;
}

static NSString *WMLibraryPath(WMOptions *options) {
    if (options.library) return options.library;
    return [WMExecutablePath().stringByDeletingLastPathComponent stringByAppendingPathComponent:@"wmref.metallib"];
}

#pragma mark - Devices

// Device specifications:
//   default            MTLCreateSystemDefaultDevice()
//   name:NAME          the device with this name (macOS: MTLCopyAllDevices)
//   provider:PATH      dlopen PATH and call WMRefProvideDevice()
//   provider:PATH:SYM  dlopen PATH and call SYM()
// A provider function has the type `id<MTLDevice> (*)(void)` and returns a
// device it keeps alive for the life of the process.
typedef id<MTLDevice> (*WMDeviceProvider)(void);

static id<MTLDevice> WMOpenDevice(NSString *spec) {
    if ([spec isEqualToString:@"default"]) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) WMDie(@"MTLCreateSystemDefaultDevice returned nil");
        return device;
    }
    if ([spec hasPrefix:@"name:"]) {
#if TARGET_OS_OSX
        NSString *name = [spec substringFromIndex:5];
        for (id<MTLDevice> device in MTLCopyAllDevices())
            if ([device.name isEqualToString:name]) return device;
        WMDie(@"no Metal device is named %@", name);
#else
        WMDie(@"name: device specifications need MTLCopyAllDevices (macOS)");
#endif
    }
    if ([spec hasPrefix:@"provider:"]) {
        NSString *rest = [spec substringFromIndex:9];
        NSString *symbol = @"WMRefProvideDevice";
        NSRange colon = [rest rangeOfString:@":" options:NSBackwardsSearch];
        if (colon.location != NSNotFound && colon.location > 0) {
            symbol = [rest substringFromIndex:colon.location + 1];
            rest = [rest substringToIndex:colon.location];
        }
        void *handle = dlopen(rest.fileSystemRepresentation, RTLD_NOW | RTLD_LOCAL);
        if (!handle) WMDie(@"dlopen %@: %s", rest, dlerror());
        WMDeviceProvider provider = (WMDeviceProvider)dlsym(handle, symbol.UTF8String);
        if (!provider) WMDie(@"%@ has no symbol %@", rest, symbol);
        id<MTLDevice> device = provider();
        if (!device) WMDie(@"%@ in %@ returned nil", symbol, rest);
        return device;
    }
    WMDie(@"unknown device specification %@", spec);
}

#pragma mark - Pretty JSON

// Containers at a depth below `compactDepth` are written one element per
// line with two-space indentation; deeper values use canonical JSON.
static void WMAppendPretty(NSMutableString *out, id value, NSUInteger depth, NSUInteger compactDepth) {
    BOOL isArray = [value isKindOfClass:[NSArray class]];
    BOOL isObject = [value isKindOfClass:[NSDictionary class]];
    if ((!isArray && !isObject) || depth >= compactDepth || [value count] == 0) {
        [out appendString:WMRefJSON(value)];
        return;
    }
    NSString *inner = [@"" stringByPaddingToLength:(depth + 1) * 2 withString:@" " startingAtIndex:0];
    NSString *outer = [@"" stringByPaddingToLength:depth * 2 withString:@" " startingAtIndex:0];
    [out appendString:isArray ? @"[\n" : @"{\n"];
    NSArray *keys = isObject ? [[value allKeys] sortedArrayUsingSelector:@selector(compare:)] : nil;
    NSUInteger count = [value count];
    for (NSUInteger i = 0; i < count; i++) {
        [out appendString:inner];
        id element;
        if (isObject) {
            [out appendString:WMRefJSON(keys[i])];
            [out appendString:@": "];
            element = value[keys[i]];
        } else {
            element = value[i];
        }
        WMAppendPretty(out, element, depth + 1, compactDepth);
        [out appendString:i + 1 < count ? @",\n" : @"\n"];
    }
    [out appendString:outer];
    [out appendString:isArray ? @"]" : @"}"];
}

static NSString *WMPretty(id value, NSUInteger compactDepth) {
    NSMutableString *out = [NSMutableString string];
    WMAppendPretty(out, value, 0, compactDepth);
    [out appendString:@"\n"];
    return out;
}

static void WMWriteFile(NSString *path, NSString *text) {
    NSError *error = nil;
    if (![text writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:&error])
        WMDie(@"cannot write %@: %@", path, error);
}

static NSString *WMSHA256(NSData *data) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *hex = [NSMutableString string];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [hex appendFormat:@"%02x", digest[i]];
    return hex;
}

#pragma mark - Child process

static int gRecordFD = 1;
static char gSignalRecord[NSIG][64];
static size_t gSignalRecordLength[NSIG];

static void WMWriteAll(int fd, const char *bytes, size_t length) {
    while (length) {
        ssize_t written = write(fd, bytes, length);
        if (written < 0) {
            if (errno == EINTR) continue;
            return;
        }
        bytes += written;
        length -= (size_t)written;
    }
}

// A terminating signal is recorded on the record stream and the process
// exits with 128 + signal, so the parent sees which signal ended the case
// without the system writing a crash report for an expected Metal failure.
static void WMChildSignal(int signal) {
    if (signal > 0 && signal < NSIG) WMWriteAll(gRecordFD, gSignalRecord[signal], gSignalRecordLength[signal]);
    _exit(128 + signal);
}

static void WMInstallSignalHandlers(void) {
    const int signals[] = {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGTRAP, SIGFPE, SIGSYS};
    for (size_t i = 0; i < sizeof(signals) / sizeof(signals[0]); i++) {
        int signal = signals[i];
        int length = snprintf(gSignalRecord[signal], sizeof gSignalRecord[signal],
                              "{\"kind\":\"signal\",\"signal\":%d}\n", signal);
        gSignalRecordLength[signal] = (size_t)length;
        struct sigaction action = {0};
        action.sa_handler = WMChildSignal;
        sigemptyset(&action.sa_mask);
        sigaction(signal, &action, NULL);
    }
}

static int WMRunInProcess(WMOptions *options, int fd) {
    if (!options.caseIdentifier) WMDie(@"--case is required");
    gRecordFD = fd;
    WMInstallSignalHandlers();
    id<MTLDevice> device = WMOpenDevice(options.device);
    NSURL *library = [NSURL fileURLWithPath:WMLibraryPath(options)];
    WMRefRunCase(device, library, options.caseIdentifier.UTF8String, ^(NSString *line) {
        NSData *data = [[line stringByAppendingString:@"\n"] dataUsingEncoding:NSUTF8StringEncoding];
        WMWriteAll(fd, data.bytes, data.length);
    });
    return 0;
}

#pragma mark - Parent

typedef struct {
    NSData *records;
    NSData *output;
    int status;
    BOOL timedOut;
} WMChildResult;

static double WMNow(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Environment for a variant: the parent's environment without any MTL_ or
// METAL_ variable, plus the variant's own variables.
static NSDictionary<NSString *, NSString *> *WMVariantVariables(NSString *variant) {
    if ([variant isEqualToString:@"debug-layer"]) return @{@"MTL_DEBUG_LAYER": @"1"};
    return @{};
}

static char **WMVariantEnvironment(NSString *variant) {
    NSMutableArray<NSString *> *entries = [NSMutableArray array];
    for (char **p = environ; *p; p++) {
        NSString *entry = @(*p);
        if ([entry hasPrefix:@"MTL_"] || [entry hasPrefix:@"METAL_"]) continue;
        [entries addObject:entry];
    }
    [WMVariantVariables(variant) enumerateKeysAndObjectsUsingBlock:^(NSString *key, NSString *value, BOOL *stop) {
        [entries addObject:[NSString stringWithFormat:@"%@=%@", key, value]];
    }];
    char **environment = calloc(entries.count + 1, sizeof(char *));
    for (NSUInteger i = 0; i < entries.count; i++) environment[i] = strdup(entries[i].UTF8String);
    return environment;
}

static int WMHighFD(int fd) {
    int high = fcntl(fd, F_DUPFD_CLOEXEC, 16);
    if (high < 0) WMDie(@"fcntl: %s", strerror(errno));
    close(fd);
    return high;
}

static WMChildResult WMSpawnChild(NSString *executable, NSArray<NSString *> *arguments, NSString *variant,
                                  double timeout) {
    int recordPipe[2], outputPipe[2];
    if (pipe(recordPipe) || pipe(outputPipe)) WMDie(@"pipe: %s", strerror(errno));
    for (int i = 0; i < 2; i++) {
        recordPipe[i] = WMHighFD(recordPipe[i]);
        outputPipe[i] = WMHighFD(outputPipe[i]);
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, outputPipe[1], 1);
    posix_spawn_file_actions_adddup2(&actions, outputPipe[1], 2);
    posix_spawn_file_actions_adddup2(&actions, recordPipe[1], 3);
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_CLOEXEC_DEFAULT);

    NSMutableArray<NSString *> *all = [NSMutableArray arrayWithObject:executable];
    [all addObjectsFromArray:arguments];
    char **argv = calloc(all.count + 1, sizeof(char *));
    for (NSUInteger i = 0; i < all.count; i++) argv[i] = strdup(all[i].UTF8String);
    char **envp = WMVariantEnvironment(variant);

    pid_t pid;
    int spawnError = posix_spawn(&pid, executable.fileSystemRepresentation, &actions, &attributes, argv, envp);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);
    for (char **p = argv; *p; p++) free(*p);
    for (char **p = envp; *p; p++) free(*p);
    free(argv);
    free(envp);
    close(recordPipe[1]);
    close(outputPipe[1]);
    if (spawnError) WMDie(@"posix_spawn %@: %s", executable, strerror(spawnError));

    NSMutableData *records = [NSMutableData data], *output = [NSMutableData data];
    struct pollfd fds[2] = {{recordPipe[0], POLLIN, 0}, {outputPipe[0], POLLIN, 0}};
    NSMutableData *sinks[2] = {records, output};
    int open = 2;
    BOOL timedOut = NO;
    double deadline = WMNow() + timeout;
    while (open > 0) {
        double remaining = deadline - WMNow();
        if (remaining <= 0) {
            timedOut = YES;
            kill(pid, SIGKILL);
            break;
        }
        int ready = poll(fds, 2, (int)MIN(remaining * 1000 + 1, 1000));
        if (ready < 0 && errno != EINTR) WMDie(@"poll: %s", strerror(errno));
        for (int i = 0; i < 2; i++) {
            if (fds[i].fd < 0 || !(fds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            char buffer[65536];
            ssize_t n = read(fds[i].fd, buffer, sizeof buffer);
            if (n > 0) {
                [sinks[i] appendBytes:buffer length:(NSUInteger)n];
            } else if (n == 0 || errno != EINTR) {
                close(fds[i].fd);
                fds[i].fd = -1;
                open--;
            }
        }
    }
    for (int i = 0; i < 2; i++)
        if (fds[i].fd >= 0) close(fds[i].fd);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    return (WMChildResult){records, output, status, timedOut};
}

// The first Metal assertion in the child's output, from the start of its
// line to its end. Validation assertions quote their message
// ("failed assertion `...'"), which may span lines and contain quotes; it
// ends at a quote followed by a line end or the end of the output. Other
// assertions ("failed assertion <condition> at line N in <method>") end at
// the line end. Object addresses (hexadecimal numbers of 9 or more digits)
// are replaced by "0x".
static NSString *WMAssertion(NSData *output) {
    NSString *text = [[NSString alloc] initWithData:output encoding:NSUTF8StringEncoding];
    if (!text) text = [[NSString alloc] initWithData:output encoding:NSISOLatin1StringEncoding];
    NSRange marker = [text rangeOfString:@"failed assertion"];
    if (marker.location == NSNotFound) return nil;
    NSRange line = [text lineRangeForRange:NSMakeRange(marker.location, 0)];
    NSUInteger start = line.location;
    NSUInteger end = NSMaxRange(line);
    if (end > start && [text characterAtIndex:end - 1] == '\n') end--;
    if ([text rangeOfString:@"failed assertion `" options:NSAnchoredSearch
                      range:NSMakeRange(marker.location, text.length - marker.location)].location != NSNotFound) {
        end = text.length;
        for (NSUInteger i = NSMaxRange(marker) + 2; i < text.length; i++) {
            if ([text characterAtIndex:i] != '\'') continue;
            if (i + 1 == text.length || [text characterAtIndex:i + 1] == '\n') {
                end = i + 1;
                break;
            }
        }
    }
    NSString *assertion = [text substringWithRange:NSMakeRange(start, end - start)];
    // Drop a system log prefix ("2026-01-02 03:04:05.678 wmref[pid:tid] ").
    NSRegularExpression *prefix = [NSRegularExpression
        regularExpressionWithPattern:@"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:.]+ [^ \\[]+\\[[0-9]+:[0-9a-fA-F]+\\] "
                             options:0 error:NULL];
    assertion = [prefix stringByReplacingMatchesInString:assertion options:0
                                                   range:NSMakeRange(0, assertion.length) withTemplate:@""];
    NSRegularExpression *address = [NSRegularExpression regularExpressionWithPattern:@"0x[0-9a-fA-F]{9,}"
                                                                             options:0 error:NULL];
    return [address stringByReplacingMatchesInString:assertion options:0
                                               range:NSMakeRange(0, assertion.length) withTemplate:@"0x"];
}

static NSDictionary *WMRunOne(NSString *executable, WMOptions *options, NSString *library, WMRefCaseInfo info,
                              NSString *variant, NSString *output, NSMutableArray *summary) {
    NSArray *arguments = @[@"child", @"--case", @(info.identifier), @"--device", options.device,
                           @"--library", library];
    WMChildResult result = WMSpawnChild(executable, arguments, variant, options.timeout);

    NSMutableArray *records = [NSMutableArray array];
    NSString *status = nil, *message = nil;
    NSNumber *signal = nil;
    NSString *text = [[NSString alloc] initWithData:result.records encoding:NSUTF8StringEncoding];
    for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
        if (line.length == 0) continue;
        NSDictionary *record = [NSJSONSerialization JSONObjectWithData:[line dataUsingEncoding:NSUTF8StringEncoding]
                                                               options:0 error:NULL];
        if (![record isKindOfClass:[NSDictionary class]] || ![record[@"kind"] isKindOfClass:[NSString class]]) {
            status = @"harness-error";
            message = [NSString stringWithFormat:@"malformed record: %@", line];
            break;
        }
        if ([record[@"kind"] isEqualToString:@"signal"]) {
            signal = record[@"signal"];
        } else if ([record[@"kind"] isEqualToString:@"end"]) {
            status = record[@"status"];
            message = record[@"message"];
        } else {
            [records addObject:record];
        }
    }
    if (!status) status = @"terminated";

    NSMutableDictionary *outcome = [NSMutableDictionary dictionary];
    if (result.timedOut) {
        outcome[@"kind"] = @"timeout";
    } else if (signal) {
        outcome[@"kind"] = @"signal";
        outcome[@"signal"] = signal;
    } else if (WIFSIGNALED(result.status)) {
        outcome[@"kind"] = @"signal";
        outcome[@"signal"] = @(WTERMSIG(result.status));
    } else {
        outcome[@"kind"] = @"exit";
        outcome[@"code"] = @(WEXITSTATUS(result.status));
    }
    NSString *assertion = WMAssertion(result.output);
    if (assertion) outcome[@"assertion"] = assertion;

    NSMutableDictionary *file = [@{@"format": WMRefFormatVersion, @"case": @(info.identifier),
                                   @"area": @(info.area), @"title": @(info.title), @"variant": variant,
                                   @"status": status, @"outcome": outcome, @"records": records} mutableCopy];
    if (message) file[@"message"] = message;
    NSString *directory = [output stringByAppendingPathComponent:variant];
    NSString *path = [directory stringByAppendingPathComponent:[@(info.identifier) stringByAppendingString:@".json"]];
    NSString *body = WMPretty(file, 2);
    WMWriteFile(path, body);
    if (options.logs) {
        NSString *log = [options.logs stringByAppendingPathComponent:
                         [NSString stringWithFormat:@"%s.%@.log", info.identifier, variant]];
        [result.output writeToFile:log atomically:YES];
    }
    NSString *kind = outcome[@"kind"];
    if ([kind isEqualToString:@"exit"]) kind = [NSString stringWithFormat:@"exit %@", outcome[@"code"]];
    else if ([kind isEqualToString:@"signal"]) kind = [NSString stringWithFormat:@"signal %@", outcome[@"signal"]];
    [summary addObject:@{@"case": @(info.identifier), @"variant": variant, @"status": status, @"outcome": kind,
                         @"sha256": WMSHA256([body dataUsingEncoding:NSUTF8StringEncoding])}];
    printf("%-13s %-11s %-10s %s\n", status.UTF8String, variant.UTF8String, kind.UTF8String, info.identifier);
    fflush(stdout);
    return file;
}

static int WMRun(WMOptions *options) {
    if (!options.output) WMDie(@"--out is required");
    NSFileManager *fm = NSFileManager.defaultManager;
    NSArray *existing = [fm contentsOfDirectoryAtPath:options.output error:NULL];
    if (existing.count) WMDie(@"%@ is not empty", options.output);
    for (NSString *variant in options.variants)
        if (![fm createDirectoryAtPath:[options.output stringByAppendingPathComponent:variant]
           withIntermediateDirectories:YES attributes:nil error:NULL])
            WMDie(@"cannot create %@", options.output);
    if (options.logs && ![fm createDirectoryAtPath:options.logs withIntermediateDirectories:YES attributes:nil error:NULL])
        WMDie(@"cannot create %@", options.logs);

    NSString *executable = WMExecutablePath();
    NSString *library = WMLibraryPath(options);
    if (![fm fileExistsAtPath:library]) WMDie(@"shader library %@ does not exist", library);
    id<MTLDevice> device = WMOpenDevice(options.device);
    NSDictionary *identity = WMRefDeviceIdentity(device);
    NSDictionary *host = WMRefHostIdentity();

    NSMutableArray *summary = [NSMutableArray array];
    NSUInteger selected = 0;
    for (NSUInteger i = 0; i < WMRefCaseCount(); i++) {
        WMRefCaseInfo info = WMRefCaseAtIndex(i);
        if (options.only.count) {
            BOOL match = NO;
            for (NSString *prefix in options.only) match |= [@(info.identifier) hasPrefix:prefix];
            if (!match) continue;
        }
        selected++;
        for (NSString *variant in options.variants) {
            @autoreleasepool {
                WMRunOne(executable, options, library, info, variant, options.output, summary);
            }
        }
    }
    if (!selected) WMDie(@"no case matches --only");

    NSMutableDictionary *variants = [NSMutableDictionary dictionary];
    for (NSString *variant in options.variants) variants[variant] = WMVariantVariables(variant);
    NSMutableDictionary *deviceRecord = [identity mutableCopy];
    deviceRecord[@"key"] = WMRefDeviceKey(identity, host);
    deviceRecord[@"source"] = options.device;
    NSDictionary *manifest = @{@"format": WMRefFormatVersion, @"build": WMRefBuildInfo(), @"device": deviceRecord,
                               @"host": host, @"variants": variants, @"cases": summary};
    WMWriteFile([options.output stringByAppendingPathComponent:@"run.json"], WMPretty(manifest, 2));
    return 0;
}

static int WMInfo(WMOptions *options) {
    id<MTLDevice> device = WMOpenDevice(options.device);
    NSDictionary *identity = WMRefDeviceIdentity(device);
    NSDictionary *host = WMRefHostIdentity();
    NSMutableDictionary *deviceRecord = [identity mutableCopy];
    deviceRecord[@"key"] = WMRefDeviceKey(identity, host);
    printf("%s", WMPretty(@{@"build": WMRefBuildInfo(), @"device": deviceRecord, @"host": host}, 3).UTF8String);
    return 0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        WMOptions *options = WMParseOptions(argc, argv);
        if ([options.command isEqualToString:@"list"]) {
            for (NSUInteger i = 0; i < WMRefCaseCount(); i++) {
                WMRefCaseInfo info = WMRefCaseAtIndex(i);
                printf("%s\t%s\t%s\n", info.identifier, info.area, info.title);
            }
            return 0;
        }
        if ([options.command isEqualToString:@"info"]) return WMInfo(options);
        if ([options.command isEqualToString:@"case"]) return WMRunInProcess(options, 1);
        if ([options.command isEqualToString:@"child"]) {
            if (fcntl(3, F_GETFD) < 0) WMDie(@"child mode expects the record stream on descriptor 3");
            return WMRunInProcess(options, 3);
        }
        if ([options.command isEqualToString:@"run"]) return WMRun(options);
        WMDie(@"unknown command %@", options.command);
    }
}
