from loguru import logger

def stat_parameters_num(model):
    trainable_num = 0
    untrainable_num = 0

    for param in model.parameters():
        if param.requires_grad:
            trainable_num += param.numel()
        else:
            untrainable_num += param.numel()
    def format_num(num):
        if num > 1e9:
            main_num = num//1e9
            s = 'B'
        elif num > 1e6:
            main_num = num//1e6
            s = 'M'
        else:
            main_num = num//1e3
            s = 'K'
        return f'{main_num}{s}'

    logger.info(f'trainable params: {format_num(trainable_num)}')
    logger.info(f'untrainable params: {format_num(untrainable_num)}')
    logger.info(f'total params: {format_num(trainable_num+untrainable_num)}')
