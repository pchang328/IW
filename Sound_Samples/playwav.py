import pyaudio
import wave
import sys

def play(file_name): 
    # length of data to read.
    # chunk = 1024
    chunk = 1024

    '''
    ************************************************************************
          This is the start of the "minimum needed to read a wave"
    ************************************************************************
    '''
    # open the file for reading.
    wf = wave.open(file_name, 'rb')

    # create an audio object
    p = pyaudio.PyAudio()

    # open stream based on the wave object which has been input.
    stream = p.open(format =
                    p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)

    # read data (based on the chunk size)
    data = wf.readframes(chunk)

    # play stream (looping from beginning of file to the end)
    while data != '':
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(chunk)

    # cleanup stuff.
    stream.close()    
    p.terminate()