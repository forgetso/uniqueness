'use strict';
import React, {Component} from 'react';

class CorrectionsList extends Component {
    constructor(props) {
        super(props);
        Object.keys(this.props.corrections).map((word), {this.setState({word:false})});

        this.setState({})
    }

    listItems = function (original_word, corrections) {
        console.log(original_word);
        console.log(corrections);
        return (
            corrections.map((correction, index) =>
                <li key={correction.toString()}>
                    <button type="button" className="btn btn-primary btn-lg btn-block"
                            onClick={(event) => this.handleClick(event, original_word, correction)}>{correction}</button>
                </li>
            )
        )
    };

    handleClick = (event, original_word, correction) => {
        console.log(event.target.innerText);
        this.props.sendSpellingCorrection(this.props.response_id, original_word, correction);
    };

    reloadComponent = (event) => {
        this.props.reloadComponent()
    };

    render() {
        return (
            <div>
                {
                    Object.keys(this.props.corrections).map((word) =>
                        <div key={word}>
                            <span className="display-4">By {word}, did you mean one of the following?</span>
                            <ul className="list-group display-2">
                                {this.listItems(word, this.props.corrections[word])}
                                <li key="none_of_the_above">
                                    <button type="button" className="btn btn-secondary btn-lg btn-block"
                                            onClick={this.reloadComponent}>None of the above
                                    </button>
                                </li>
                            </ul>
                        </div>
                    )
                }
            </div>
        )
    }
}


export default CorrectionsList;
