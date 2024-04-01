#ifndef ROUTES_H
#define ROUTES_H

#include "cgraph_queries.h"
#include "server.h"

nlohmann::json route_command(std::string & command, nlohmann::json & data, SharedMemory &sharedMemory) {

    ffml_cgraph * cgraph = sharedMemory.cgraph;

    if (command == "ECHO") {
        return data;
    }

    else if (command == "REVERSE") {
        // assuming data is string
        std::string s = data;
        std::reverse(s.begin(), s.end());
        return s;
    }

    else if (command == "QUIT") {
        exit(0);
    }

    else if (command == "PAUSE") {
        sharedMemory.should_pause = true;
        return "OK";
    }

    else if (command == "RESUME") {
        sharedMemory.should_pause = false;
        return "OK";
    }

    else if (command == "CGRAPH_INFO") {
        return cgraph_info(cgraph);
    }

    else if (command == "CGRAPH_TENSOR_DATA") {
        return cgraph_tensor_data(cgraph, data["key"]);
    }

    // else if (command == "CGRAPH_TENSOR_GRAD") {
    //     return cgraph_tensor_data(cgraph, data["key"], true);
    // }

    else {
        return "ERROR: command not found";
    }

}

#endif