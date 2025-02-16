### Alive

### Install packages
Install required packages for AR to work.
```shell
pip install -r requirements.txt
```

### Upload image and video for vector db

```shell
curl -X POST http://127.0.0.1:5000/upload \
  -F 'image=@/path/to/your/image.jpg' \
  -F 'video=@/path/to/your/video.mp4'
```
There are two scripts i.e. vector_genedration.py is used to populate the vector db and image_search.py is used to search the 
encoded image from db.


### Client-Server architecture
Client-server code added for AR. It's slow due to sending encoded images over the network for real-time augmentation.
```shell
python server.py
python client.py
```

https://alive-frontend-omega.vercel.app/
