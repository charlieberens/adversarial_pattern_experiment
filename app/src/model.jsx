import React, { Component } from "react";
import * as tf from "@tensorflow/tfjs";
import "./styles/model.scss";

const classes = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel",
];

export class Model extends Component {
    constructor(props) {
        super(props);

        this.state = {};
    }

    setImage = (image) => {
        this.setState({ image });
    };

    runModel = async () => {
        const img = new Image();
        img.src = URL.createObjectURL(this.state.image);
        const model = await tf.loadLayersModel(
            `${process.env.PUBLIC_URL}/models/reinforced_full_30/model.json`
        );

        const img_tensor = tf.image.resizeBilinear(
            tf.browser.fromPixels(img),
            [256, 256]
        );
        const prediction = model.predict(img_tensor.expandDims());
        let output = {};
        prediction.dataSync().forEach((val, index) => {
            if (val) {
                output[classes[index]] = val;
            }
        });
        console.log(prediction.dataSync());
        this.setState({ guess: output });
    };

    render() {
        return (
            <div className="modelCont">
                <div className="modelLeft">
                    <div className="modelUploadCont">
                        <h3>Upload Image</h3>
                        <input
                            type="file"
                            name="inputImage"
                            onChange={(e) => {
                                this.setImage(e.target.files[0]);
                            }}
                        />
                        {this.state.image && (
                            <div>
                                <img
                                    className="displayImage"
                                    alt=""
                                    src={URL.createObjectURL(this.state.image)}
                                />
                            </div>
                        )}
                    </div>
                    {this.state.image && (
                        <button
                            className="identifyButton"
                            onClick={this.runModel}
                        >
                            Identify
                        </button>
                    )}
                </div>
                {this.state.guess && (
                    <div className="guess">
                        {Object.entries(this.state.guess).map(
                            ([classification, val]) => (
                                <h2 key={classification}>
                                    {classification}:{" "}
                                    {val.toString().includes("e")
                                        ? val
                                              .toString()
                                              .split("e")[0]
                                              .substring(0, 8) +
                                          "e" +
                                          val.toString().split("e")[1]
                                        : val.toString().substring(0, 8)}
                                </h2>
                            )
                        )}
                    </div>
                )}
            </div>
        );
    }
}

export default Model;
