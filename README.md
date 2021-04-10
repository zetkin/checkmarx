Document Scanner Stack
======================

Services:
 * `checkmarx`: Find checked boxes in a document
 * `metadata-server`: Serve example metadata on a document
 * `frontengels`: React app front-end for `checkmarx`


Usage:

```
docker-compose up -d --build
```

Then go to ~~`localhost:3000`~~ (front-end currently not working)
`localhost:5000/docs` and upload an image or submit an HTTP request to
`localhost:5000/scan` with a form parameter `image` containing the image to
process.
