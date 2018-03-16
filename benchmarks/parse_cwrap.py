import yaml

if __name__ == "__main__":
  all_text = ""
  with open("/private/home/cpuhrsch/dev/pytorch/aten/src/ATen/Declarations.cwrap", "r") as f:
    all_text = f.read()
  all_decl = all_text.split("]]\n[[")[1:]
  query_arguments = ['THTensor* self', 'THTensor* self']
  good_decls = []
  for decl in all_decl:
    try:
      y = yaml.load(decl)
    except:
      continue
    if 'arguments' in y and y['arguments'] == query_arguments:
      if 'name' in y and y['name'][-1] == '_':
        if 'types' in y and y['types'] == ['floating_point']:
          good_decls += [y]

  for y in good_decls:
    print(y)
#  for l in sorted(good_decls):
#    print(l)
