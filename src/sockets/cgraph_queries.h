#ifndef CGRAPH_QUERIES_H
#define CGRAPH_QUERIES_H

#include <string>
#include "../ffml/ffml.h"

#include <nlohmann/json.hpp>

nlohmann::json cgraph_info(ffml_cgraph * cgraph) {
    nlohmann::json nodes = nlohmann::json::array();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ffml_tensor * t = cgraph->nodes[i];

        nlohmann::json::object_t node_json = {
            {"key", t->key},
            {"name", t->name},
            {"n_dims", t->n_dims},
            {"ne", { t->ne[0], t->ne[1], t->ne[2], t->ne[3] }},
            {"nb", { t->nb[0], t->nb[1], t->nb[2], t->nb[3] }},
            {"broadcast_dim_src0", { t->broadcast_dim_src0[0], t->broadcast_dim_src0[1], t->broadcast_dim_src0[2], t->broadcast_dim_src0[3] }},
            {"broadcast_dim_src1", { t->broadcast_dim_src1[0], t->broadcast_dim_src1[1], t->broadcast_dim_src1[2], t->broadcast_dim_src1[3] }},
            {"broadcast_to_src0_enabled", { t->broadcast_to_src0_enabled[0], t->broadcast_to_src0_enabled[1], t->broadcast_to_src0_enabled[2], t->broadcast_to_src0_enabled[3] }},
            {"broadcast_to_src1_enabled", { t->broadcast_to_src1_enabled[0], t->broadcast_to_src1_enabled[1], t->broadcast_to_src1_enabled[2], t->broadcast_to_src1_enabled[3] }},
            {"op", t->op},
            {"src0", (t->src0 && ffml_get_tensor_by_key(cgraph, t->src0->key)) ? t->src0->key : -1},
            {"src1", (t->src1 && ffml_get_tensor_by_key(cgraph, t->src1->key)) ? t->src1->key : -1},
        };

        nodes.push_back(node_json);
    }

    nlohmann::json j;
    j["cgraph_info"]["nodes"] = nodes;
     
    return j;
}

nlohmann::json _get_cgraph_tensor_data(ffml_cgraph * cgraph, ffml_tensor * t, bool grad = false) {
    auto data_part = nlohmann::json::array();

    // use ffml_get_data_flat
    uint64_t n_elements = t->nelem;
    for (uint64_t i = 0; i < n_elements; i++) {
        float d = (grad) ? ffml_get_grad_flat(t, i) : ffml_get_data_flat(t, i);
        // printf("d: %f\n", d);
        data_part.push_back(d);
    }

    return data_part;
}

nlohmann::json cgraph_tensor_data(ffml_cgraph * cgraph, uint64_t key) {
    ffml_tensor * t = ffml_get_tensor_by_key(cgraph, key);

    if (t == nullptr) {
        return "ERROR: tensor not found";
    }

    nlohmann::json j;
    j["cgraph_tensor_data"]["key"] = t->key;


    j["cgraph_tensor_data"]["data"] = _get_cgraph_tensor_data(cgraph, t, false).dump();
    j["cgraph_tensor_data"]["grad"] = _get_cgraph_tensor_data(cgraph, t, true).dump();

    return j;
}

#endif