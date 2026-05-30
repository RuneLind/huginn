import os
import shutil
import pickle
import tempfile


class CorruptArtifactError(Exception):
    """Raised when an on-disk binary artifact cannot be deserialized.

    Names the offending file and the underlying cause so a truncated or corrupt
    index surfaces a clear message instead of a bare UnpicklingError/EOFError.
    """


class DiskPersister:
    TEMP_PREFIX = ".tmp-"

    def __init__(self, base_path):
        self.base_path = base_path
        # honour the process umask so atomically-written files keep the same
        # permissions an in-place open(path, 'w') would have produced
        current_umask = os.umask(0o022)
        os.umask(current_umask)
        self._file_mode = 0o666 & ~current_umask

    def save_text_file(self, data, file_path):
        self.__atomic_write(file_path, lambda file: file.write(data.encode("utf-8")))

    def read_text_file(self, file_path):
        path = os.path.join(self.base_path, file_path)

        with open(path, 'r', encoding="utf-8") as file:
            return file.read()

    def save_bin_file(self, data, file_path):
        self.__atomic_write(file_path, lambda file: pickle.dump(data, file))

    def read_bin_file(self, file_path):
        path = os.path.join(self.base_path, file_path)

        with open(path, 'rb') as file:
            try:
                return pickle.load(file)
            except (pickle.UnpicklingError, EOFError, ValueError, OSError) as e:
                raise CorruptArtifactError(
                    f"Failed to deserialize '{path}': the file is truncated or "
                    f"corrupt ({type(e).__name__}: {e}). Re-index the collection."
                ) from e

    def atomic_write_set(self):
        """Stage a set of files, then commit them together.

        Returns a context manager: add files via add_text_file/add_bin_file inside
        a `with` block. Every file is written to a fsynced temp first; only once the
        block exits without error are they renamed into place (a sequence of
        metadata-only os.replace calls). If the block raises, the staged temps are
        discarded and the existing target files are left untouched, so a crash or
        error mid-write can never leave a half-written, mutually-inconsistent set.
        """
        return _AtomicWriteSet(self)

    def create_folder(self, folder_name):
        directory_path = os.path.join(self.base_path, folder_name)
        os.makedirs(directory_path)

    def remove_folder(self, folder_name):
        directory_path = os.path.join(self.base_path, folder_name)

        if os.path.exists(directory_path):
            shutil.rmtree(directory_path, ignore_errors=True)

    def remove_file(self, file_path):
        path = os.path.join(self.base_path, file_path)

        if os.path.exists(path):
            os.remove(path)

    def is_path_exists(self, relative_path):
        path = os.path.join(self.base_path, relative_path)
        return os.path.exists(path)

    def read_folder_files(self, relative_path):
        path = os.path.join(self.base_path, relative_path)
        files = []
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                if filename.startswith(self.TEMP_PREFIX):
                    # in-flight temp from an interrupted atomic write — not a real artifact
                    continue
                files.append(os.path.relpath(os.path.join(root, filename), path))
        return files

    def __atomic_write(self, file_path, write_fn):
        target = os.path.join(self.base_path, file_path)
        temp = self._write_temp(target, write_fn)
        try:
            os.replace(temp, target)
        except BaseException:
            self._discard_temp(temp)
            raise
        self._fsync_dir(os.path.dirname(target))

    def _write_temp(self, target, write_fn):
        """Write durably to a uniquely-named temp file beside target and return its
        path. The caller renames it into place (now, or once a whole set is staged)."""
        self.__make_sure_path_exists(target)

        directory = os.path.dirname(target)
        fd, temp = tempfile.mkstemp(dir=directory, prefix=self.TEMP_PREFIX)
        try:
            os.fchmod(fd, self._file_mode)
            with os.fdopen(fd, "wb") as temp_file:
                write_fn(temp_file)
                temp_file.flush()
                os.fsync(temp_file.fileno())
        except BaseException:
            self._discard_temp(temp)
            raise
        return temp

    def _discard_temp(self, temp):
        if os.path.exists(temp):
            os.remove(temp)

    def _fsync_dir(self, directory):
        if not directory:
            return
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            # not all platforms allow fsync on a directory handle
            pass
        finally:
            os.close(dir_fd)

    def __make_sure_path_exists(self, path):
        directory_path = os.path.dirname(path)

        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path)


class _AtomicWriteSet:
    def __init__(self, persister):
        self._persister = persister
        self._staged = []  # (temp_path, target_path)

    def add_text_file(self, data, file_path):
        self.__stage(file_path, lambda file: file.write(data.encode("utf-8")))

    def add_bin_file(self, data, file_path):
        self.__stage(file_path, lambda file: pickle.dump(data, file))

    def __stage(self, file_path, write_fn):
        target = os.path.join(self._persister.base_path, file_path)
        temp = self._persister._write_temp(target, write_fn)
        self._staged.append((temp, target))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            for temp, _ in self._staged:
                self._persister._discard_temp(temp)
            return False

        directories = set()
        for temp, target in self._staged:
            os.replace(temp, target)
            directories.add(os.path.dirname(target))
        for directory in directories:
            self._persister._fsync_dir(directory)
        return False
