from .Client import DriverClient, WorkerClients
from .MatrixHandle import MatrixHandle
from .Parameter import Parameter
import time
import h5py
import os
import importlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class AlchemistSession:

    driver = []
    workers = []
    libraries = dict()

    workers_connected = False

    def __init__(self):
        print("Starting Alchemist session ... ", end="", flush=True)
        self.driver = DriverClient()
        self.workers = WorkerClients()
        self.workers_connected = False
        print("ready")

    def __del__(self):
        print("Ending Alchemist session")
        self.close()

    def namestr(self, obj, namespace):
        return [name for name in namespace if namespace[name] is obj]

    def read_from_hdf5(self, filename):
        print("Loaded " + filename)
        return h5py.File(filename, 'r')
    
    def get_number_of_chunks(self, size_of_input_dataset, size_of_chunks):
        if (size_of_input_dataset % size_of_chunks > 0):
            number_of_chunks = int(size_of_input_dataset / size_of_chunks) + 1
        else:
            number_of_chunks = int(size_of_input_dataset / size_of_chunks)
        return number_of_chunks
    
    def send_dask_matrix(self, matrix, print_times=False, layout="MC_MR"):
        max_block_rows = 100
        max_block_cols = 20000
        
        size_of_row_chunks = matrix.chunksize[0]
        size_of_col_chunks = matrix.chunksize[1]
        size_of_chunks = size_of_row_chunks * size_of_col_chunks
        
        size_of_matrix = matrix.size
        
        print("Size of row chunks:",size_of_row_chunks)
        print("Size of col chunks:",size_of_col_chunks)
        print("Size of chunks:",size_of_chunks)
        print("Size of input matrix:",size_of_matrix)
        
        no_of_chunks = self.get_number_of_chunks(size_of_matrix, size_of_chunks)
        
        print("Number of total chunks:",no_of_chunks)
        
        row_dim = matrix.shape[0]
        col_dim = matrix.shape[1]
        
        print("Row size:",row_dim)
        print("Column size:",col_dim)
        
        no_of_row_chunks = int(row_dim / size_of_row_chunks)
        no_of_col_chunks = int(col_dim / size_of_col_chunks)
        
        print("Number of row chunks:",no_of_row_chunks)
        print("Number of column chunks:",no_of_col_chunks)
        
        matrix_name = matrix.name
        matrix_layers = matrix.dask.layers[matrix_name]

        for slice_row_id in range(no_of_row_chunks):
            for slice_col_id in range(no_of_col_chunks):
                print("slice_row_id",slice_row_id)
                print("slice_col_id",slice_col_id)
                startIndex_row = slice_row_id*size_of_row_chunks
                endIndex_row = slice_row_id*size_of_row_chunks + size_of_row_chunks
                startIndex_column = slice_col_id*size_of_col_chunks
                endIndex_column = slice_col_id*size_of_col_chunks + size_of_col_chunks
                print()
                chunk_index = matrix_layers[(matrix_name,slice_row_id, slice_col_id)]
                print("printing the layer:",chunk_index)
                print()
                current_chunk = chunk_index[0](matrix_layers[chunk_index[1]],chunk_index[2])
                print("printing the required array:")
                print(current_chunk)
                # instead of printing the current chunk, call send_dask_chunk here.
                print()
                print("start row index", startIndex_row)
                print("start column index", startIndex_column)
                if (size_of_matrix % size_of_chunks > 0):
                    endIndex_row = size_of_matrix
                    endIndex_column = size_of_matrix
                print("end row index", endIndex_row)
                print("end col index", endIndex_column)

    def send_dask_chunk(self, dask_chunk, print_times=False, layout="MC_MR"):
        max_block_rows = 100
        max_block_cols = 20000
        # to do- priya - change the current send_dask_chunks signature to send the startIndex_row, startIndex_column, endIndex_row and endIndex_column for each chunk
        # Make necessary changes at alchemist end to re-construct the matrix  based on the above values for each chunk received
        
        (num_rows, num_cols) = dask_chunk.shape
        
        print("Sending array info to Alchemist ... ", end="", flush=True)
        start = time.time()
        ah = self.get_matrix_handle(dask_chunk, layout=layout)
        end = time.time()
        print("done ({0:.4e}s)".format(end - start))
        
        print("Sending array data to Alchemist ... ", end="", flush=True)
        start = time.time()
        times = self.workers.send_matrix_blocks(ah, dask_chunk)
        end = time.time()
        print("done ({0:.4e}s)".format(end - start))
        if print_times:
            self.print_times(times, name=ah.name)
        #     self.driver.send_block(mh, block)
        return ah

    def send_matrix(self, matrix, print_times=False, layout="MC_MR"):
        max_block_rows = 100
        max_block_cols = 20000

        (num_rows, num_cols) = matrix.shape

        print("Sending array info to Alchemist ... ", end="", flush=True)
        start = time.time()
        ah = self.get_matrix_handle(matrix, layout=layout)
        end = time.time()
        print("done ({0:.4e}s)".format(end - start))

        print("Sending array data to Alchemist ... ", end="", flush=True)
        start = time.time()
        times = self.workers.send_matrix_blocks(ah, matrix)
        end = time.time()
        print("done ({0:.4e}s)".format(end - start))
        if print_times:
            self.print_times(times, name=ah.name)
        #     self.driver.send_block(mh, block)

        return ah

    def fetch_matrix(self, mh, print_times=False):

        matrix = np.zeros((mh.num_rows, mh.num_cols))

        print("Fetching data for array {0} from Alchemist ... ".format(mh.name), end="", flush=True)
        start = time.time()
        matrix, times = self.workers.get_matrix_blocks(mh, matrix)
        end = time.time()
        print("done ({0:.4e}s)".format(end - start))
        if print_times:
            self.print_times(times, name=mh.name)
        return matrix

    def print_times(self, times, name=" ", spacing="  "):
        print("")
        if name is "":
            print("Data transfer times breakdown")
        else:
            print("Data transfer times breakdown for array {}".format(name))
        print("{}---------------------------------------------------------------------------------------------------------------".format(spacing))
        print("{}  Worker  |   Serialization time   |       Send time        |      Receive time      |  Deserialization time  ".format(spacing))
        print("{}---------------------------------------------------------------------------------------------------------------".format(spacing))
        for i in range(self.workers.num_workers):
            print("{0}    {1:3d}   |       {2:.4e}       |       {3:.4e}       |       {4:.4e}       |       {5:.4e}       ".format(spacing, i+1, times[0, i], times[1, i], times[2, i], times[3, i]))
        print("{}---------------------------------------------------------------------------------------------------------------".format(spacing))
        print("")

    def send_hdf5(self, f):

        sh = f.shape

        num_rows = sh[0]
        num_cols = sh[1]

        mh = self.get_array_handle(f)

        chunk = 1000

        for i in range(0, num_rows, chunk):
            self.workers.send_blocks(mh, np.float64(f[i:min(num_rows, i+chunk), :]), i)

        return mh

    def get_matrix_handle(self, data=[], name="", sparse=0, layout="MC_MR"):
        # print("Sending matrix info to Alchemist ... ", end="", flush=True)
        # start = time.time()
        (num_rows, num_cols) = data.shape

        ah = self.driver.send_matrix_info(name, num_rows, num_cols, sparse, MatrixHandle.layouts[layout])
        # end = time.time()
        # print("done ({0:.4e})".format(end - start))
        return ah

    def load_library(self, name, path=""):
        if self.workers_connected:
            lib_id = self.driver.load_library(name, path)
            if lib_id <= 0:
                print("ERROR: Unable to load library \'{name}\' at {path}, check path.".format(name=name, path=path))
                return 0
            else:
                module = importlib.import_module("alchemist.lib." + name + "." + name)
                library = getattr(module, name)()

                # msg = 'The {module_name} module has the following methods: {methods}'
                # print(msg.format(module_name=name, methods=dir(library)))

                library.set_id(lib_id)
                library.set_alchemist_session(self)

                self.libraries[lib_id] = library

                print("Library \'{name}\' at {path} successfully loaded.".format(name=name, path=path))
                return library

    def run_task(self, lib_id, name, in_args):
        print("Alchemist started task '" + name + "' ... ", end="", flush=True)
        start = time.time()
        out_args = self.driver.run_task(lib_id, name, in_args)
        end = time.time()
        print("done ({0:.4e}s)".format(end - start))
        return out_args

    def display_parameters(self, parameters, preamble="", spacing="    "):

        if len(preamble) > 0:
            print(preamble)
        for key, p in parameters.items():
            print(spacing + p.to_string())
        # for key, value in parameters.items():
        #     dt_name = ""
        #     for name, code in Parameter.datatypes.items():
        #         if code == value.datatype:
        #             dt_name = name
        #     print(spacing + key + " = " + str(value.value) + " (" + dt_name + ")")

    def connect_to_alchemist(self, address, port):
        self.driver.address = address
        self.driver.port = port

        self.driver.connect()

    def send_test_string(self):
        self.driver.send_test_string()

    def request_test_string(self):
        self.driver.request_test_string()

    def list_available_libraries(self):
        self.driver.list_available_libraries()

    def convert_hdf5_to_parquet(self, h5_file, parquet_file, chunksize=100000):

        stream = pd.read_hdf(h5_file, chunksize=chunksize)

        for i, chunk in enumerate(stream):
            print("Chunk {}".format(i))

            if i == 0:
                # Infer schema and open parquet file on first chunk
                parquet_schema = pa.Table.from_pandas(df=chunk).schema
                parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='snappy')

            table = pa.Table.from_pandas(chunk, schema=parquet_schema)
            parquet_writer.write_table(table)

        parquet_writer.close()

    def load_from_hdf5(self, file_name, dataset_name):
        return self.driver.load_from_hdf5(file_name, dataset_name)

    def get_matrix_info(self):
        self.driver.get_matrix_info()

    def request_workers(self, num_requested_workers):
        self.workers.add_workers(self.driver.request_workers(num_requested_workers))
        self.workers.print()
        self.workers_connected = self.workers.connect()

    def yield_workers(self, yielded_workers=[]):
        deallocated_workers = self.driver.yield_workers(yielded_workers)
        if len(deallocated_workers) == 0:
            print("No workers were deallocated")
        else:
            s = ""
            if len(deallocated_workers) > 1:
                s = "s"
            print("Listing {0} deallocated Alchemist worker{1}:".format(len(deallocated_workers), s))
            self.workers.print(deallocated_workers)

    def list_alchemist_workers(self):
        all_workers = self.driver.list_all_workers()
        if len(all_workers) == 0:
            print("No Alchemist workers")
        else:
            s = ""
            if len(all_workers) > 1:
                s = "s"
            print("Listing {0} Alchemist worker{1}:".format(len(all_workers), s))
            self.workers.print(all_workers)

    def list_all_workers(self):
        all_workers = self.driver.list_all_workers()
        if len(all_workers) == 0:
            print("No Alchemist workers")
        else:
            s = ""
            if len(all_workers) > 1:
                s = "s"
            print("Listing {0} Alchemist worker{1}:".format(len(all_workers), s))
            self.workers.print(all_workers)

    def list_active_workers(self):
        active_workers = self.driver.list_active_workers()
        if len(active_workers) == 0:
            print("No active Alchemist workers")
        else:
            s = ""
            if len(active_workers) > 1:
                s = "s"
            print("Listing {0} active Alchemist worker{1}:".format(len(active_workers), s))
            self.workers.print(active_workers)

    def list_inactive_workers(self):
        inactive_workers = self.driver.list_inactive_workers()
        if len(inactive_workers) == 0:
            print("No inactive Alchemist workers")
        else:
            s = ""
            if len(inactive_workers) > 1:
                s = "s"
            print("Listing {0} inactive Alchemist worker{1}:".format(len(inactive_workers), s))
            self.workers.print(inactive_workers)

    def list_assigned_workers(self):
        assigned_workers = self.driver.list_assigned_workers()
        if len(assigned_workers) == 0:
            print("No assigned Alchemist workers")
        else:
            s = ""
            if len(assigned_workers) > 1:
                s = "s"
            print("Listing {0} assigned Alchemist worker{1}:".format(len(assigned_workers), s))
            self.workers.print(assigned_workers)

    def stop(self):
        self.close()

    def close(self):
        self.driver.close()
        self.workers.close()



