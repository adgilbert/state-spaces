optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "src.statespaces.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "timm_cosine": "src.statespaces.utils.optim.schedulers.TimmCosineLRScheduler",
}

model = {
    # Backbones from this repo
    "model": "src.statespaces.models.sequence.SequenceModel",
    "unet": "src.statespaces.models.sequence.SequenceUNet",
    "sashimi": "src.statespaces.models.sequence.sashimi.Sashimi",
    "sashimi_standalone": "sashimi.sashimi.Sashimi",
    # Baseline RNNs
    "lstm": "src.statespaces.models.baselines.lstm.TorchLSTM",
    "gru": "src.statespaces.models.baselines.gru.TorchGRU",
    "unicornn": "src.statespaces.models.baselines.unicornn.UnICORNN",
    "odelstm": "src.statespaces.models.baselines.odelstm.ODELSTM",
    "lipschitzrnn": "src.statespaces.models.baselines.lipschitzrnn.RnnModels",
    "stackedrnn": "src.statespaces.models.baselines.samplernn.StackedRNN",
    "stackedrnn_baseline": "src.statespaces.models.baselines.samplernn.StackedRNNBaseline",
    "samplernn": "src.statespaces.models.baselines.samplernn.SampleRNN",
    # Baseline CNNs
    "ckconv": "src.statespaces.models.baselines.ckconv.ClassificationCKCNN",
    "wavegan": "src.statespaces.models.baselines.wavegan.WaveGANDiscriminator", # DEPRECATED
    "wavenet": "src.statespaces.models.baselines.wavenet.WaveNetModel",
    "torch/resnet2d": "src.statespaces.models.baselines.resnet.TorchVisionResnet",
    # Nonaka 1D CNN baselines
    "nonaka/resnet18": "src.statespaces.models.baselines.nonaka.resnet.resnet1d18",
    "nonaka/inception": "src.statespaces.models.baselines.nonaka.inception.inception1d",
    "nonaka/xresnet50": "src.statespaces.models.baselines.nonaka.xresnet.xresnet1d50",
}

layer = {
    "id": "src.statespaces.models.sequence.base.SequenceIdentity",
    "lstm": "src.statespaces.models.sequence.rnns.lstm.TorchLSTM",
    "sru": "src.statespaces.models.sequence.rnns.sru.SRURNN",
    "lssl": "src.statespaces.models.sequence.ss.lssl.LSSL",
    "s4": "src.statespaces.models.sequence.ss.s4.S4",
    "standalone": "src.statespaces.models.s4.s4.S4",
    "s4d": "src.statespaces.models.s4.s4d.S4D",
    "ff": "src.statespaces.models.sequence.ff.FF",
    "rnn": "src.statespaces.models.sequence.rnns.rnn.RNN",
    "mha": "src.statespaces.models.sequence.mha.MultiheadAttention",
    "conv1d": "src.statespaces.models.sequence.convs.conv1d.Conv1d",
    "conv2d": "src.statespaces.models.sequence.convs.conv2d.Conv2d",
    "performer": "src.statespaces.models.sequence.attention.linear.Performer",
    "mega": "src.statespaces.models.sequence.mega.MegaBlock",
}

callbacks = {
    "timer": "src.statespaces.callbacks.timer.Timer",
    "params": "src.statespaces.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing": "src.statespaces.callbacks.progressive_resizing.ProgressiveResizing",
}
