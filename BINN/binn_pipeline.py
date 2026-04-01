"""
- Load data
- Create loaders
- Init BINN
- Feed data:
    # Train BINN
    # Test BINN
- Evaluate + Interpret BINN
- Compare to SVM
"""

import argparse
import os
from Binn import BINN

def pipeline(
        to_include: list, 
        train_size: int,

    ) -> None:
        
        

        print('Pipeline completed')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Run annData objects through BINN pipeline"
    )

    # Positional argument: accepts one or more integers
    parser.add_argument(
        "to_include",
        type=int,        
        nargs='+',       
        help="indices to include: \n0=astro \n1=exc1 \n2=exc2 \n3=exc3 \n4=immune \n5=inhi \n6=oligo \n7=opcs \n8=vasc",
    )


