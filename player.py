import httplib2
import math
import numpy as np
import queue
import time
import threading

import help_functions as hf


class Player(object):
    """
    A Python-based PC player
    ...
    Methods
    -------
    run()
        Run the PC player
    """

    def __init__(self, host, port, buffer_size, gop, fps, n_seg, data,
                 rate_adapter, n_conn, pc_names, pc_ids):
        """
        Parameters
        ----------
        host : str
            IP address
        port : int
            Port number
        buffer_size : float
            Buffer size [s]
        gop : int
            Group of pictures length
        fps : int
            Frames per second
        n_seg : int
            Number of video segments
        data : dataframe
            Data all segments, point cloud objects and quality representations
        rate_adapter : RateAdaptation
            Rate adaptation heuristic
        n_conn : int
            Number of parallel TCP connections
        pc_ids : list
            List of the PC IDs
        pc_names : dict
            Names of the PC objects
        """

        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.gop = gop
        self.fps = fps
        self.n_seg = n_seg
        self.data = data
        self.rate_adapter = rate_adapter
        self.n_conn = n_conn
        self.pc_ids = pc_ids
        self.pc_names = pc_names

        self.download_queue = DownloadQueue()
        self.seg_queue = queue.Queue()
        self.time_start_playing = -1
        self.frame_played = 0

        self.freeze_freq = 0
        self.freeze_dur = 0

    def _play(self):
        """Simulates playout of video content
        """

        # Run until the video has finished playing
        n_played = 0
        freezing = False

        while n_played < self.n_seg:
            # Check if content is available
            if n_played > 0 and self.seg_queue.qsize() == 0:
                freezing = True

            # Get segment from queue
            qualities = self.seg_queue.get()
            time_curr = time.time()
            if n_played == 0:
                self.time_start_playing = time_curr

            # Update freeze statistics
            if freezing:
                print("Freeze of length %f observed for segment %i" %
                      (time_curr - time_last_played - self.gop / self.fps,
                       n_played + 1))
                self.freeze_freq += 1
                self.freeze_dur += time_curr - time_last_played - self.gop / self.fps

            # Pass one segment duration, while updating the frame iterator
            time_now = time.time()
            time_remaining = max(self.gop / self.fps - (time_now - time_curr), 0)
            time_it = time_now
            while time_it - time_now < time_remaining:
                self.frame_played = math.trunc(n_played * self.gop + (time.time() - time_curr) * self.fps)
                #print(f'Set played frame to {self.frame_played}')
                time.sleep(0.001)
                time_it = time.time()

            print(f'Now at {time_it - self.time_start_playing}')

            # End current task
            self.seg_queue.task_done()
            time_last_played = time_curr
            freezing = False
            n_played += 1

    def _do_work(self):
        """Initiates a new HTTP connection and retrieves video content
        """

        # Initiate new HTTP connection
        h = httplib2.Http()

        while True:
            # Retrieve segment number, point cloud number and quality
            tup = self.download_queue.get()
            s_id, pc_id, quality = tup[1], tup[2], tup[3]

            # If segment number equals 0: stop running
            if s_id == 0:
                self.download_queue.task_done()
                return

            # Log the quality
            self.qualities[self.pc_ids.index(pc_id)] = quality

            # Generate URL and send the request
            url = self._generate_url(s_id, pc_id, quality)
            print(url)
            #(resp_headers, content) = h.request(url, "GET")
            time.sleep(0.15)

            # End task
            self.download_queue.task_done()

    def _generate_url(self, s_id, pc_id, quality):
        """Generates URL based on segment number, tile number and quality
        Parameters
        ----------
        s_id : int
            Segment number
        pc_id : int
            Point cloud number
        quality : int
            Quality representation
        Returns
        -------
        str
            Generated URL
        """

        pc_name = self.pc_names[pc_id]
        h = f'http://{self.host}:{self.port}'
        d = f'{pc_name}/{self.gop}/{quality}'
        f = f'{pc_name}_{s_id:04d}.bin'
        return f'{h}/{d}/{f}'

    def _buffer(self):
        """Buffers VR content
        """

        # Initialize settings
        bandwidth = 0

        # Start workers
        for _ in range(self.n_conn):
            t = threading.Thread(target=self._do_work)
            t.daemon = True
            t.start()

        # Loop over all segments, in order
        for s_id in range(1, self.n_seg + 1):

            # Frame information
            frame_start = (s_id - 1) * self.gop + 1

            print(f'Buffering segment {s_id}, containing frames {frame_start} through {frame_start + self.gop - 1}')

            # Reset downloaded qualities
            self.qualities = [0] * len(self.pc_names)

            # Set time since last update
            time_last_update = time.time()

            # If playout has not started yet, default values are assigned
            if self.time_start_playing < 0:
                time_dl = 0

            # Determine bitrate budget
            budget_bits = bandwidth * time_dl

            # Rate adaptation
            qualities = self.rate_adapter.adapt(frame_start, self.gop,
                                                self.frame_played, budget_bits,
                                                self.data)
            print(qualities)

            # Time before downloading current segment
            time_start = time.time()

            # Send required resources to download queue
            d = 0
            for i in range(len(self.pc_ids)):
                pc_id = self.pc_ids[i]
                quality = qualities[i]
                self.download_queue.put((d, s_id, pc_id, quality))
                d += 1

            # Wait until all tiles are downloaded
            self.download_queue.join()

            print(self.qualities)

            # Total download time
            time_passed = time.time() - time_start

            # Determine bandwidth
            bits = 0
            for i in range(len(self.pc_ids)):
                pc_id = self.pc_ids[i]
                quality = self.qualities[i]
                f = self.data.loc[(self.data['frame'] >= frame_start) & (self.data['frame'] < frame_start + self.gop)]
                f = f.groupby(by=['pc'], as_index=False)
                f = f.mean()
                f = f.loc[f['pc'] == pc_id]
                bits += f.iloc[0]['r%i' % quality] * self.gop
            bandwidth = bits / time_passed
            print(bandwidth / 1000000, time_passed)

            # Push segment to playout queue
            self.seg_queue.put(self.qualities[:])

            print(self.seg_queue.qsize(), self.buffer_size, self.fps, self.gop, self.buffer_size * self.fps / self.gop)

            # If the buffer is not full, continue
            if self.seg_queue.qsize() < self.buffer_size * self.fps / self.gop:

                print('Buffer not full!')

                frames_buffered = frame_start + self.gop - 1 - self.frame_played

                print(f'Frames buffered: {frames_buffered}')

                time_dl = max((frames_buffered + self.gop) / self.fps - self.buffer_size, 0)

            # Otherwise, wait till a buffered segment is consumed
            else:

                print('Buffer full, waiting...')

                limit = self.buffer_size * self.fps / self.gop
                while self.seg_queue.qsize() >= limit:
                    time.sleep(0.001)
                time_dl = self.gop / self.fps

            print("----------------------------------------------------------")

        # Terminate workers
        for _ in range(self.n_conn):
            self.download_queue.put((0, 0, 0, 0))

    def run(self):
        """Run the PC player
        """

        t = threading.Thread(target=self._play)
        t.daemon = True
        t.start()
        self._buffer()
        self.seg_queue.join()
        t.join()


class DownloadQueue:
    """
    A queue for prioritized PC downloads
    ...
    Methods
    -------
    put(tup)
        Put a tuple in the queue
    get()
        Get the next tuple in the queue
    replace(tups)
        Replace tuples, possibly containing new decisions
    qsize()
        Number of tuples remaining
    task_done()
        Indicate finishing a task
    join()
        Wait for the queue to be empty
    """

    def __init__(self):
        self.d = {}
        self.i = 0
        self.mutex = threading.Lock()

    def put(self, tup):
        self.mutex.acquire()
        self.d[tup[2]] = tup
        self.i += 1
        self.mutex.release()

    def get(self):
        while True:
            self.mutex.acquire()
            if len(self.d) > 0:
                k = min(self.d.items(), key=lambda x: x[1][0])[0]
                tup = self.d[k]
                del self.d[k]
                self.mutex.release()
                return tup
            self.mutex.release()

    def replace(self, tups):
        self.mutex.acquire()
        for tup in tups:
            if tup[2] in self.d:
                self.d[tup[2]] = tup
        self.mutex.release()

    def qsize(self):
        return len(self.d)

    def task_done(self):
        self.mutex.acquire()
        self.i -= 1
        self.mutex.release()

    def join(self):
        while self.i > 0:
            time.sleep(0.001)
