import argparse
from hparams import hparams, hparams_debug_string
import os
import io
from tacotron.synthesize import tacotron_synthesize
from tacotron.synthesizer import Synthesizer as tacatron_Synthesizer
from wavernn_vocoder.synthesize import wavernn_synthesize
from datasets import audio
import web
import hashlib

class SynthesisResource:
  def on_get(self, req, res):
    if not req.params.get('text'):
      raise falcon.HTTPBadRequest()
    mels, speaker_ids = tacatron_synthesizer.synthesize([req.params.get('text')],['test.npy'], None, None, None, True)
    assert len(mels) == 1
    wav = audio.inv_mel_spectrogram(mels[0].T, hparams)
    out = io.BytesIO()
    audio.save_wav(wav, out, hparams.sample_rate)
    res.data = out.getvalue()
    res.content_type = 'audio/wav'


tacatron_synthesizer = tacatron_Synthesizer()

class Handle(object):
    def GET(self):
        print('dbdb')
        try:
            data = web.input()
            if len(data) == 0:
                return "hello, this is handle view"
            signature = data.signature
            timestamp = data.timestamp
            nonce = data.nonce
            echostr = data.echostr
            token = "wuwuyueyuexinxin" 

            list = [token, timestamp, nonce]
            list.sort()
            sha1 = hashlib.sha1()
            map(sha1.update, list)
            hashcode = sha1.hexdigest()
            print("handle/GET func: hashcode, signature: ", hashcode, signature)
#            if hashcode == signature:
            return echostr
 #           else:
  #              return ""
        except Exception as Argument:
            return Argument


urls = (
    '/wx','Handle',
    '/', 'Handle'
)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
#  parser.add_argument('--tacotron_checkpoint', default='/home/wuyuexin333/TTS-System/logs-Tacotron/taco_pretrained/tacotron_model.ckpt-24000', help='Full path to model checkpoint')
#  parser.add_argument('--wavernn_checkpoint', help='Full path to model checkpoint')
  # parser.add_argument('--output_dir', help='')
#  parser.add_argument('--port', type=int, default=80)
#  parser.add_argument('--hparams', default='',
#    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
#  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 # hparams.parse(args.hparams)
  # print(hparams_debug_string())
  tacotron_checkpoint = '/home/wuyuexin333/TTS-System/logs-Tacotron/taco_pretrained/tacotron_model.ckpt-24000'
  wavernn_checkpoint = ''
  tacatron_synthesizer.load(tacotron_checkpoint, hparams)
#  if (args.wavernn_checkpoint):
#    pass

  app = web.application(urls, globals())
  app.run()
  # print('Serving on port %d' % args.port)
  # simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
