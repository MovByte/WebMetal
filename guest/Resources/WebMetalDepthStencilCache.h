// The device-wide depth-stencil state cache of the A10X driver: AGX::Device keeps an os_unfair_lock at +0x800 and a
// std::map at +0x808 from a state's packed descriptor and label to the state object, which
// -[AGXA10FamilyDevice newDepthStencilStateWithDescriptor:] (AGXMetalA10 0x21c8d53e0) returns for an equal descriptor.
// WebMetalDevice holds one (guest/Plugin/WebMetalDevice.mm); WebMetalDepthStencilState.mm uses it.

#pragma once

#import <Foundation/Foundation.h>

#include <os/lock.h>

#include <cstdint>
#include <map>
#include <string>
#include <utility>

namespace webmetal {

struct DepthStencilCache {
    struct Entry {
        uint32_t loadStoreNeeds = 0;  // the map value's needs word, copied into the state
        id __strong state = nil;      // the cache's reference: the driver never removes an entry
    };
    os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
    std::map<std::pair<uint64_t, std::string>, Entry> states;
};

}  // namespace webmetal
