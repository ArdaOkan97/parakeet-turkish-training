import argparse
from py_spy_nonblocking_bounded_helper_v2 import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);a=p.parse_args()
    run('toy',a.ROOT_review,__file__)
