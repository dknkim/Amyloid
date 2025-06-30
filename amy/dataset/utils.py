

T1_MPRAGE = ["MPRAGE", "MP-RAGE", "IR-SPGR", "FSPGR", "MP_RAGE", "mprage"]
FLAIR = ["FLAIR", "t2_flair"]
T2 = ["T2_TSE", "T2_STAR", "T2-Star", "T2-TSE", "T2-weighted", "PD_T2", "PD-T2", "T2__FGRE", "T2_FSE", "T2star", "T2", "t2_tse", "t2_blade"]

PET_TAU = ["AV1451", "Tau", "TAU"]

AMYLOID = ["AV45", "FDG", "PIB", "FBB", "AV-45"]
amyloid_lower = ["florbetaben", "florbetapir", "f18", "flortaucipir"]


def sort_sequence(file_name):
    
    for m in T1_MPRAGE:
        if m in file_name:
            return "MPRAGE"
    for f in FLAIR:
        if f in file_name:
            return "FLAIR"
    for t in T2:
        if t in file_name and "flair" not in file_name.lower():
            return "T2"
    
    for p in PET_TAU:
        if p in file_name:
            return "TAU"
    
    for a in AMYLOID:
        if a in file_name:
            return "AMYLOID"
    for a in amyloid_lower:
        if a in file_name.lower():
            return "AMYLOID"
    
    return "OTHER"