def generate_all_combinations_aux(list_dict, orig_dict, aux_dict, rest_list):
    if len(aux_dict) < len(orig_dict):
        key = list(orig_dict)[len(aux_dict)]
        if isinstance(orig_dict[key], list):
            if not orig_dict[key] or (key in rest_list and 
                not isinstance(orig_dict[key][0], list)):
                aux_dict[key] = orig_dict[key]
                generate_all_combinations_aux(list_dict, 
                                              orig_dict, 
                                              aux_dict, 
                                              rest_list)
            else:
                for item in orig_dict[key]:
                    if isinstance(item, dict):
                        new_list_dict = []
                        new_aux_dict = {}
                        generate_all_combinations_aux(new_list_dict, 
                                                    item, 
                                                    new_aux_dict, 
                                                    rest_list)
                    else:
                        new_list_dict = [item]
                    for new_dict in new_list_dict:
                        new_aux_dict = aux_dict.copy()
                        new_aux_dict[key] = new_dict
                        
                        generate_all_combinations_aux(list_dict,
                                                    orig_dict, 
                                                    new_aux_dict, 
                                                    rest_list)
        elif isinstance(orig_dict[key], dict):
            new_list_dict = []
            new_aux_dict = {}
            generate_all_combinations_aux(new_list_dict, 
                                          orig_dict[key], 
                                          new_aux_dict, 
                                          rest_list)
            for dictionary in new_list_dict:
                new_aux_dict = aux_dict.copy()
                new_aux_dict[key] = dictionary
                generate_all_combinations_aux(list_dict, 
                                              orig_dict, 
                                              new_aux_dict, 
                                              rest_list)
        else:
            aux_dict[key] = orig_dict[key]
            generate_all_combinations_aux(list_dict, 
                                          orig_dict, 
                                          aux_dict, 
                                          rest_list)
    else:
        list_dict.append(aux_dict)


def generate_all_combinations(original_dict, restriction_list):
    list_dict = []
    aux_dict = {}
    generate_all_combinations_aux(list_dict, original_dict, aux_dict, restriction_list)
    return list_dict