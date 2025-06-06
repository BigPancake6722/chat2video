OnlyTrainParams = list(
    [
        "background_type",
    ]
)

ModelParams = dict(
    max_points = 75000,
    part = "all",
    background_type = "black",
)


ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 3,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64]
    },
    multires = [1,2],
    defor_depth = 2,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False,
    only_infer=False,
    d_model=64,
    n_head=2,
    drop_prob=0.2,
    ffn_hidden=128,
    n_layer=1,
    train_tri_plane=True, 
)
OptimizationParams = dict(
    position_lr_init = 1.6e-4,
    grid_lr_init = 1.6e-3,
    dataloader=False,
    densify_from_iter =1000,
    densification_interval = 100,
    iterations = 10000,
    coarse_iterations = 7999,
    densify_until_iter = 7000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    densify_grad_threshold_coarse = 0.001,
    lip_fine_tuning = True,
    depth_fine_tuning = True,
    deformation_lr_final = 0.00001,
    deformation_lr_init = 0.0001,
    split_gs_in_fine_stage=False,
    canonical_tri_plane_factor_list=["opacity","shs"], #["scales","rotations","opacity","shs"]
    add_point=False,
    attn_reg_from_iter=3000,
    coarse_surface_reg_from_iter=3000,
)