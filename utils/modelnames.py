from models.lp1_simplified import LP1Simplified
from models.lp1_sumtoone import LP1SumToOne
from models.plrec import plrec


models = {
    "PLRec": plrec,
}

critiquing_models = {
    "LP1Simplified": LP1Simplified,
    "LP1SumToOne": LP1SumToOne
}
