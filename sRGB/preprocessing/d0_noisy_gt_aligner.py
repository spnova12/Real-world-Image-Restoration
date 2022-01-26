import argparse

import d0_noisy_gt_aligner.noisy_gt_aligner_train as aligner

# parser = argparse.ArgumentParser(description='Argparse Tutorial')
#
# parser.add_argument('--train', '-train', action='store_true')
# parser.add_argument('--test', '-test', action='store_true')
#
# args = parser.parse_args()
#
# if args.train and not args.test:
#     d1_rain.rain_train.main()
#
# if not args.train and args.test:
#     d1_rain.rain_test.main()
#

aligner.main()