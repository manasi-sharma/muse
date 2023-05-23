from attrdict import AttrDict as d

from muse.envs.pymunk.push_t_lang import PushTEnv

lang_mode_to_dim = {'voltron': 4, 'clip': 512, 't5': 768, 't5_sentence': 768, 'distilbert': 768, 'distilbert_sentence': 768}

export = PushTEnv.default_params & d(
    cls=PushTEnv,
    lang_dim= lang_mode_to_dim["voltron"]
)
