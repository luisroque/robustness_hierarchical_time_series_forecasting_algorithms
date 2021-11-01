from run_create_dataset_versions import CreateNewVersion

# Create new versions of a dataset using time series augmentation methods
# CreateNewVersion('prison').create_new_version_random()

# Create new version of a dataset using same transformation per version
CreateNewVersion('prison').create_new_version_single_transf()
