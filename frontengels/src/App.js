import React from 'react';
import axios from 'axios';
import logo from './img/logo.png';
import Webcam from "react-webcam";
import './App.css';

function b64toBlob(byteCharacters, contentType, sliceSize) {
    contentType = contentType || '';
    sliceSize = sliceSize || 512;

    var byteCharacters2 = byteCharacters.slice("data:image/jpeg;base64,".length)
    var byteCharacters3 = atob(byteCharacters2);
    var byteArrays = [];

    for (var offset = 0; offset < byteCharacters3.length; offset += sliceSize) {
        var slice = byteCharacters3.slice(offset, offset + sliceSize);

        var byteNumbers = new Array(slice.length);
        for (var i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
        }

        var byteArray = new Uint8Array(byteNumbers);

        byteArrays.push(byteArray);
    }

    var blob = new Blob(byteArrays, {
        type: contentType
    });
    return blob;
}

class App extends React.Component {

        constructor(props) {
            super(props);
            this.state = {
                picture: null,
                result: null,
            };
            this.webcamRef = React.createRef();
        }

        render() {
                const videoConstraints = {
                    width: 1280,
                    height: 720,
                    facingMode: "environment"
                };
                const capture = () => {
                    const imageSrc = this.webcamRef.current.getScreenshot();
                    this.setState({
                        picture: imageSrc
                    });
                };
                const paragraphs = [];
                if (this.state.result) {
                    if (this.state.result.length) {
                        for (var i = 0; i < this.state.result.length; i++) {
                            paragraphs.push( < p key = {
                                    this.state.result[i]
                                } > {
                                    this.state.result[i]
                                }
                                was Marxed < /p>);
                            }
                        } else {
                            paragraphs.push( < p > No result found < /p>)
                            }
                        }
                        const image = this.state.picture ?
                            <
                            img src = {
                                this.state.picture
                            }
                        alt = "webcam capture" / >: < Webcam
                        audio = {
                            false
                        }
                        ref = {
                            this.webcamRef
                        }
                        screenshotFormat = "image/jpeg"
                        width = "100%"
                        videoConstraints = {
                            videoConstraints
                        }
                        justifyContent = "left" /
                            >
                            const scan = () => {
                                var form = new FormData();
                                form.append("image", b64toBlob(this.state.picture));
                                axios({
                                        method: "post",
                                        url: "/scan",
                                        data: form,
                                        headers: {
                                            'Content-Type': 'multipart/form-data'
                                        }
                                    })
                                    .then((res) => {
                                        this.setState({
                                            picture: null,
                                            result: res.data.result
                                        });
                                    })
                                    .catch((err) => {
                                        console.error(err);
                                    });
                            }

                        return ( <
                            div className = "App" >
                            <
                            header className = "App-header" >
                            <
                            img src = {
                                logo
                            }
                            className = "App-logo"
                            alt = "logo" / >
                            <
                            /header> {
                                image
                            } <
                            button onClick = {
                                capture
                            } > Capture photo < /button> <
                            button onClick = {
                                scan
                            }
                            disabled = {!this.state.picture
                            } > Check Marx < /button> {
                                paragraphs
                            } <
                            /div>
                        );
                    }
                }

                export default App;
