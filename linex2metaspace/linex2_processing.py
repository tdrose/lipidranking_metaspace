import linex2 as lx2


def get_lx2_ref_lip_dict():
    """

    Args:

    Returns:
    
    """
    return lx2.parse_lipid_reference_table_dict(lx2.load_data.STANDARD_LIPID_CLASSES)


def get_lx2_ref_lips():
    """

    Args:

    Returns:
    
    """
    return lx2.parse_lipid_reference_table(lx2.load_data.STANDARD_LIPID_CLASSES)


def get_organism_combined_class_reactions(ref_lip_dict, organism='HSA'):
    """

    Args:

    Returns:
    
    """
    class_reaction_list = [
        lx2.tmp_dirty_functions.make_organism_reaction_list_from_reactome(
            lx2.load_data.REACTOME_OTHERS_UNIQUE,
            lx2.load_data.REACTOME_REACTION_CURATION,
            lx2.load_data.REACTOME_REACTION_TO_MOLECULE,
            ref_lip_dict,
            lx2.load_data.REACTOME_REACTION_DETAILS,
            verbose=False,
            organism=organism),

        lx2.tmp_dirty_functions.make_all_reaction_list_from_rhea(
            lx2.load_data.RHEA_OTHERS_UNIQUE,
            lx2.load_data.RHEA_REACTION_CURATION,
            lx2.load_data.RHEA_REACTION_TO_MOLECULE,
            ref_lip_dict,
            gn_mapping=lx2.load_data.RHEA_MAPPING,
            verbose=False)[0]
    ]
    class_reacs = lx2.utils.combine_reactions(
        class_reaction_list[0],
        class_reaction_list[1])

    return class_reacs
