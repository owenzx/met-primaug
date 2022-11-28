# prim2primX Data Augmentation

### Augmenting the training data

We provide scripts to run data augmentation on both the SCAN addprim_jump (`augment_scan_jump.sh`) and the COGS (`augment_cogs.sh`) dataset.

To control the size of augmented, you can change the parameters in the script:
* `--num_extra_actions` defines the number of mutated primitives per original primitive. For example, to 
recreate the `+ 5 prim` in our Table 7, set `--num_extra_actions=5`.
* `--num_mutation_per_example`: when augmenting COGS, it is too expensive to enumerate all possible
mutation permutations for each example since each example may contain a lot of primitives.
For example, if an example has 5 primitives and `--num_extra_actions=5`, then we can have
  (num_extra_actions+1)^5 = 7776 different mutations. In this case, we only sample `num_mutation_per_example`
mutation per example. In the paper, we set `--num_mutation_per_example=2` for COGS.

### Using the augmented data
After running the data augmentation, you will get files like `train_primx2s2.tsv` in `comp-data/COGS/cogs`
and `tasks_train_primx5_addprim_jump.txt` in `comp-data/SCAN/add_prim_split`.
In order to train the model with these augmented data, simply set
* `--special_train_data=train_primx5` for SCAN addprim_jump;
* `--special_train_data=train_primx5s10` for SCAN around_right;
* `--special_train_data=train_primx2s2` for COGS. 

