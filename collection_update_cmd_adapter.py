import argparse

from main.utils.logger import setup_root_logger
from main.factories.update_collection_factory import create_collection_updater

setup_root_logger()

ap = argparse.ArgumentParser()
ap.add_argument("-collection", "--collection", required=True, help="Collection name (will be used to determine root folder and manifest file)")
args = vars(ap.parse_args())

create_collection_updater = create_collection_updater(args['collection'])

create_collection_updater.run()