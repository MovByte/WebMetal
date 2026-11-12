// The plugin's not-implemented channel.
//
// Everything the plugin does not implement, and every refusal to run on a Metal.framework it was not built for, is
// reported here: one line on standard error and an os_log fault, both naming the class and selector. Under strict mode
// (WEBMETAL_STRICT=1 in the process environment) the process then aborts. Otherwise the call fails: a method raises
// an NSException named WebMetalNotImplemented, an initializer returns nil. Nothing is answered in place of a result.

#pragma once

#import <Foundation/Foundation.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The name of the exception a failed call raises.
extern NSString *const WMNotImplementedException;

// Reports a problem with `what` (a method, or a step such as registration). Aborts under strict mode; returns
// otherwise, and the caller fails the operation.
void WMReport(const char *what, const char *problem);

// Reports -[class selector] (+[...] for a class) of the receiver as not implemented and fails the call. Never returns.
__attribute__((noreturn)) void WMLoudMethod(id self, SEL _cmd, const char *detail);

// As WMLoudMethod, for a query whose argument lies outside what the plugin answers.
__attribute__((noreturn)) void WMLoudArgument(id self, SEL _cmd, uint64_t argument, const char *detail);

// The implementation of every generated LOUD stub: any selector of any signature; it reads only self and _cmd.
__attribute__((noreturn)) void WMLoudStub(id self, SEL _cmd);

// The number of reports this process has made, and a copy of the most recent one (for tests).
size_t WMReportCount(void);
void WMLastReport(char *buffer, size_t size);

#ifdef __cplusplus
}
#endif
