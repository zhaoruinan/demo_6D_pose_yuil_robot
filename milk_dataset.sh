for i in {1..100}
do
        echo "generate $i x100 images now(10 000 in total)"
   blenderproc run examples/datasets/bop_object_pose_sampling/main.py ~/data_0/BOP milk ./dataset_out
done