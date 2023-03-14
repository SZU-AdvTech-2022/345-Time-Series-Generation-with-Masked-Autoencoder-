from models import *


def run_mae(args):
    # Data Loading
    # 获取数据
    ori_data = load_data(args)
    np.save(args.ori_data_dir, ori_data)  # save ori_data before normalization
    # 标准化数据
    ori_data, min_val, max_val = min_max_scalar(ori_data)

    # Write statistics 记录数据相关信息
    args.min_val = min_val
    args.max_val = max_val
    # np.var()为方差
    args.data_var = np.var(ori_data)
    print(f'{args.data_name} data variance is {args.data_var}')

    # Initialize the Model  初始化模型
    model = InterpoMAE(args, ori_data)
    if args.training:
        print(f'Start AutoEncoder Training! {args.ae_epochs} Epochs Needed.')
        model.train_ae()
        print(f'Start Embedding Training! {args.embed_epochs} Epochs Needed.')
        model.train_embed()
        print(f'Start Reconstruction Training! {args.recon_epochs} Epochs Needed.')
        model.train_recon()
        print('Training Finished!\n')
    else:
        # 加载已经训练好的模型
        model = load_model(args, model)
        print(f'Successfully loaded the model!\n')

    # Evaluation   评估
    if args.embed_epochs == 0 and args.recon_epochs == 0:
        print('\nStart Evaluate as an Auto Encoder.')
        model.evaluate_ae()
    else:
        print('\nStart Evaluate as MAE.')
        model.evaluate_random_mae()

        # print('\nStart Synthesis by Cross Masks, Results Averaged.')
        # model.synthesize_cross_average()
        # print('\nStart Synthesis by Cross Masks, Results Concatenated.')
        # model.synthesize_cross_concate()
        # print('\nStart Synthesis by Random Masks, Results Averaged.')
        # model.synthesize_random_average()
        # print(f'{args.experiment_name} Done!')
    #
    # # Calculate Predictive and Discriminative Scores
    # print('Calculating Pred and Disc Scores\n')
    # calculate_pred_disc(args)


def main():
    # 返回当前工作目录
    home = os.getcwd()
    args = load_arguments(home)
    run_mae(args)


if __name__ == '__main__':
    main()
