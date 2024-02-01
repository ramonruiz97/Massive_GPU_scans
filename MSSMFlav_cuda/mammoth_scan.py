from tools import initialize
import argparse
import os


def checker_file(**args):
  type = args["type"]
  gpu = int(args["gpu"])
  scenario = args["scenario"]
  while os.path.exists("tuples/{}_{}_{}.root".format(type,scenario, gpu)):
    gpu += 1
  return gpu


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Configurable of scanner")
  parser.add_argument( "--type", help="Gpu that you want to use", choices=["Flat", "Genetic"], default="Flat")
  parser.add_argument( "--gpu", help="Gpu that you want to use", default=0)
  parser.add_argument( "--scenario", help="Scenario", default="scenarioA")
  parser.add_argument( "--N", help="Number of points", default=100000)

  args = vars(parser.parse_args())
  initialize(int(args["gpu"]))
  gpu = checker_file(**args)
  N = int(args["N"])
  scenario = args["scenario"]
  type = args["type"]
  initialize(gpu)
  from MammothScanner import *
  m = MammothScanner(N, scenario)
  start = timer()
  if "Genetic" in type:
    m.GeneticScan("tuples/Genetic_{}_{}".format(scenario, gpu),NG = 100, f=0.5, cr = .9 )
  elif "Flat" in type:
    m.FlatScan("tuples/Flat_{}_{}".format(scenario, gpu))
  else:
    print("Type of Scanner not recognized")

  print timer() - start


