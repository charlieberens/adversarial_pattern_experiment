import "./styles/app.scss";
import React, { Component } from "react";
import Model from "./model";
import { FaGithub } from "react-icons/fa";

class App extends Component {
    constructor(props) {
        super(props);

        this.state = {
            models: [
                {
                    name: "Reinforced Full",
                    id: "reinforced_full",
                    epochs: [30],
                },
            ],
        };
    }

    handleModelChange = (e) => {
        this.setState({ model: e.target.value });
    };
    handleEpochChange = (e) => {
        this.setState({ epoch: e.target.value });
    };

    render() {
        console.log(this.state);
        return (
            <div className="app">
                <div className="dropdowns">
                    <div>
                        <h4>Model</h4>
                        <select
                            value={this.state.model}
                            onChange={this.handleModelChange}
                        >
                            {!this.state.model && (
                                <option value="select" className="select">
                                    Select
                                </option>
                            )}
                            {this.state.models.map((model) => (
                                <option key={model.id} value={model.id}>
                                    {model.name}
                                </option>
                            ))}
                        </select>
                    </div>
                    {this.state.model && (
                        <div>
                            <h4>Training Length</h4>
                            <select
                                value="select"
                                onChange={this.handleEpochChange}
                            >
                                {!this.state.epoch && (
                                    <option value="select" className="select">
                                        Select
                                    </option>
                                )}

                                {this.state.models
                                    .find(
                                        (model) => model.id === this.state.model
                                    )
                                    .epochs.map((epoch) => (
                                        <option key={epoch} value={epoch}>
                                            {epoch} Epochs
                                        </option>
                                    ))}
                            </select>
                        </div>
                    )}
                </div>
                {this.state.model && this.state.epoch && (
                    <Model model={this.state.model} epoch={this.state.epoch} />
                )}
                <a
                    className="gh"
                    href="https://github.com/charlieberens/adversarial_pattern_experiment"
                >
                    <FaGithub />
                </a>
            </div>
        );
    }
}

export default App;
