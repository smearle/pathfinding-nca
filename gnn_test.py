from pdb import set_trace as TT

from models.gnn import get_self_edges, get_neighb_edges, batch_edges


if __name__ == "__main__":
    se = get_self_edges(2, 2)
    print(se)
    ge = get_neighb_edges(2, 2)
    print(ge)

    be = batch_edges(ge, 3, 4)
    print(be)