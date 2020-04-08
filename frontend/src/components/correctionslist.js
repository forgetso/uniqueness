'use strict';
import React, {Component} from 'react';

class CorrectionsList extends Component {
    constructor(props) {
        super(props);
        this.state = {};
        console.log(Object.keys(this.props.corrections));
        Object.keys(this.props.corrections).map((word) => {
            this.setState({[word]: {show: false, submitted: false}});
        });
        this.setState({[Object.keys(this.props.corrections)[0]['show']]: true})
    }

    listItems = function (original_word, corrections, index) {
        console.log(original_word);
        console.log(corrections);
        return (
            corrections.map((correction, index) =>
                <li key={correction.toString()}>
                    <button type="button" className="btn btn-primary btn-lg btn-block"
                            onClick={(event) => this.handleClick(event, original_word, correction, index)}>{correction}</button>
                </li>
            )
        )
    };

    handleClick = (event, original_word, correction, index) => {
        console.log(event.target.innerText);
        this.props.sendSpellingCorrection(this.props.response_id, original_word, correction);
        this.setState({[[original_word]['show']]: false});
        this.setState({[[original_word]['submitted']]: true});
        let next_index = index + 1;
        let next_word = Object.keys(this.props.corrections)[next_index];
        this.setState({[[next_word]['show']]: true});
    };

    hideCorrections = (word) => {
        this.setState({[[word]['show']]: false});
        this.setState({[[word]['submitted']]: true});
        if (Object.keys(this.props.corrections).indexOf(word) === this.props.corrections.length - 1) {
            this.props.reloadComponent();
        }
    };

    render() {
        //TODO work out why this.state[mk] is not set
        return (
            <div>

                {
                    Object.keys(this.props.corrections).map(function (word, index) {
                            return this.state[word]['show'] ? (
                                    <div key={word}>
                                        <span className="display-4">By {word}, did you mean one of the following?</span>
                                        <ul className="list-group display-2">
                                            {this.listItems(word, this.props.corrections[word], index)}
                                            <li key="none_of_the_above">
                                                <button type="button" className="btn btn-secondary btn-lg btn-block"
                                                        onClick={(event) => this.hideCorrections(word)}>None of the above
                                                </button>
                                            </li>
                                        </ul>
                                    </div>)
                                : null
                        }, this
                    )
                }
            </div>
        )
    }
}


export default CorrectionsList;
