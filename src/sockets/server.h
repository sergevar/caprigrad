#ifndef SOCKET_SERVER_H
#define SOCKET_SERVER_H

#include "../ffml/ffml.h"
#include <nlohmann/json.hpp>

struct SharedMemory {
    ffml_cgraph * cgraph;
    bool should_pause = false;
};

void server_start(ffml_cgraph * cgraph);
void server_push_string(const std::string& data);
void server_push_event(const std::string eventType, const nlohmann::json& data);
void server_loop_interrupt();

#endif