import argparse

from main.utils.logger import setup_root_logger
from main.factories.update_collection_factory import create_collection_updater

setup_root_logger()

ap = argparse.ArgumentParser()
ap.add_argument("-collection", "--collection", required=True, help="Collection name (will be used to determine root folder and manifest file)")
ap.add_argument("--contextual-model", required=False, default=None,
                help="Override contextual-prefix backend spec (e.g. 'ollama:qwen3.6:35b-a3b-nvfp4', 'claude-code:claude-haiku-4-5', 'none'). "
                     "If omitted, uses manifest['contextualPrefix']['model'] — letting scheduled updates inherit the create-time choice.")
ap.add_argument("--contextual-cache", required=False, default=None,
                help="Path to the contextual-prefix cache JSON (defaults to data/contextual_caches/<name>.json — outside the collection folder so it survives re-creates).")
ap.add_argument("--contextual-workers", required=False, type=int, default=1,
                help="How many documents to prefix concurrently (default 1 = sequential).")
args = vars(ap.parse_args())

create_collection_updater = create_collection_updater(args['collection'],
                                                      contextual_backend_spec=args['contextual_model'],
                                                      contextual_cache_path=args['contextual_cache'],
                                                      contextual_workers=args['contextual_workers'])

create_collection_updater.run()
