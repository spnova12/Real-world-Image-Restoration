import argparse

import d0_raw.raw_train
# import d1_rain.rain_test

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

d0_raw.raw_train.main()