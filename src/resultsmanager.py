import os
import pandas as pd 
from pathlib import Path
import boto3
import io

def add_result(result):
		return ResultsManager().add_result(result)

class ResultsManager:

		def __init__(self):
			self.main_fname = 'Results/main.pkl'
			self.main_key = 'main.pkl'
			self.buckets = {
					'main': 'alnn-main-bucket-8nyb87yn8',
					'animations': 'alnn-animations-bucket-8nyb87yn8',
					'zipfiles': 'alnn-zipfiles-bucket-8nyb87yn8'
			}

		def add_result(self, result):
			self.add_result_local(result)
			self.zipdir_local(result)
			self.add_result_cloud(result)

		def add_result_local(self, result):
			if os.path.exists(self.main_fname):
					data = pd.read_pickle(self.main_fname)
					data = pd.concat((data, result))
					data.index = list(range(data.shape[0]))
					data.to_pickle(self.main_fname)
			else:
					result.to_pickle(self.main_fname)
			return

		def zipdir_local(self, result):
			lg = result.loc[0]['local_graphs']
			rootdir = lg['local_graphs_root']
			cmd = f"./src/cmds/zipdir.sh {rootdir}"
			os.system(cmd)
			return        

		def add_result_cloud(self, result):
			zk, ak = self.upload_zipfile_animation(result)
			result['cloud_graphs'] = [{'zip_key': zk, 'animation_key': ak}]
			main = self.download_main()
			if main is not None:
					main = pd.concat((main, result))
			else:
					main = result
			self.upload_main(main)
			return

		def s3client(self):
			client = boto3.client("s3")
			return client

		def upload_zipfile_animation(self, result):
			s3 = self.s3client()
			ts = result.loc[0]['timestamp']
			zip_fname = result.loc[0]['local_graphs']['local_graphs_root'] + '.zip'
			zip_key = f"{ts}.zip"
			anim_fname = result.loc[0]['local_graphs']['local_graphs_root'] + '/animation.mp4'
			anim_key = f"{ts}.mp4"
			s3.upload_file(zip_fname, self.buckets['zipfiles'], zip_key)
			s3.upload_file(anim_fname, self.buckets['animations'], anim_key)
			return zip_key, anim_key

		def make_public(self, bucket, key):
			res = boto3.resource("s3")
			res.Object(bucket, key).Acl().put(ACL='public-read') # pylint: disable=no-member
			return True

		def download_main(self):
			s3 = self.s3client()
			try:
				tmpfile = Path(f'/tmp/alnn_main_download_{int(time.time())}')
				tmpfile.touch()
				tmpfilepath = str(tmpfile.resolve())
				with open(tmpfilepath, 'wb') as f:
					s3.download_fileobj(self.buckets['main'], self.main_key, f)
				main = pd.read_pickle(tmpfilepath)
				tmpfile.unlink()
				return main
			except ClientError:
				return None

		def upload_main(self, main):
			s3 = self.s3client()
			main_buf = io.StringIO()
			main.to_pickle(main_buf)
			main_buf.seek(0)
			s3.upload_fileobj(main_buf, self.buckets['main'], self.main_key)
			self.make_public(self.buckets['main'], self.main_key)
			return