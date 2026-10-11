import timm

models = timm.list_models("*dino*")
#models = timm.list_models("*")

print(len(models))
print(models)
