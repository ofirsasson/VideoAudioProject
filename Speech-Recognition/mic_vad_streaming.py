import time, logging
from datetime import datetime
import threading, collections, os, os.path
import deepspeech
import queue
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
import audioclass3

logging.basicConfig(level=20)


def main(ARGS):

    # Load DeepSpeech model
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, 'output_graph.pb')
        ARGS.lm = os.path.join(model_dir, ARGS.lm)
        ARGS.trie = os.path.join(model_dir, ARGS.trie)

    print('Initializing model...')
    logging.info("ARGS.model: %s", ARGS.model)
    model = deepspeech.Model(ARGS.model, ARGS.beam_width)
    if ARGS.lm and ARGS.trie:
        logging.info("ARGS.lm: %s", ARGS.lm)
        logging.info("ARGS.trie: %s", ARGS.trie)
        model.enableDecoderWithLM(ARGS.lm, ARGS.trie, ARGS.lm_alpha, ARGS.lm_beta)

    # Start audio with VAD
    vad_audio = audioclass3.VADAudio(aggressiveness=ARGS.vad_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate,
                         file=ARGS.file)
    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()
    
    # Stream from microphone to DeepSpeech using VAD
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner='line')
    stream_context = model.createStream()
    wav_data = bytearray()
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            logging.debug("streaming frame")
            model.feedAudioContent(stream_context, np.frombuffer(frame, np.int16))
            if ARGS.savewav: wav_data.extend(frame)
        else:
            if spinner: spinner.stop()
            logging.debug("end utterence")
            text = model.finishStream(stream_context)
            if ARGS.savewav:
                vad_audio.write_wav(os.path.join(ARGS.savewav, text+".wav"), wav_data)
                wav_data = bytearray()
            
            print("Recognized: %s" % text)
            stream_context = model.createStream()

if __name__ == '__main__':
    BEAM_WIDTH = 500
    DEFAULT_SAMPLE_RATE = 16000
    LM_ALPHA = 0.75
    LM_BETA = 1.85

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-v', '--vad_aggressiveness', type=int, default=3,
                        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")
    parser.add_argument('-w', '--savewav',
                        help="Save .wav files of utterences to given directory")
    parser.add_argument('-f', '--file',
                        help="Read from .wav file instead of microphone")

    parser.add_argument('-m', '--model', required=True,
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-l', '--lm', default='lm.binary',
                        help="Path to the language model binary file. Default: lm.binary")
    parser.add_argument('-t', '--trie', default='trie',
                        help="Path to the language model trie file created with native_client/generate_trie. Default: trie")
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")
    parser.add_argument('-la', '--lm_alpha', type=float, default=LM_ALPHA,
                        help=f"The alpha hyperparameter of the CTC decoder. Language Model weight. Default: {LM_ALPHA}")
    parser.add_argument('-lb', '--lm_beta', type=float, default=LM_BETA,
                        help=f"The beta hyperparameter of the CTC decoder. Word insertion bonus. Default: {LM_BETA}")
    parser.add_argument('-bw', '--beam_width', type=int, default=BEAM_WIDTH,
                        help=f"Beam width used in the CTC decoder when building candidate transcriptions. Default: {BEAM_WIDTH}")

    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)
