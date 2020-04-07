'use strict';
import React, {Component} from 'react';

class CorrectionsList extends Component {
    constructor(props) {
        super(props);

    }

    listItems = function (corrections) {
        console.log(corrections);
        return (
            corrections.map((correction) =>
                <li key={correction.toString()}>
                    <button type="button" className="btn btn-primary btn-lg btn-block"
                            onClick={this.handleClick}>{correction}</button>
                </li>
            )
        )
    };

    handleClick = (event) => {
        event.preventDefault();
        this.props.sendSpellingCorrection(this.props.response_id, event.target.innerText);
    };

    reloadComponent = (event) => {
        this.props.reloadComponent();
    };

    render() {
        return (
            <div>
                {
                    Object.keys(this.props.corrections).map((word) =>
                        <div>
                            <span className="display-4">By {word}, did you mean one of the following?</span>
                            <ul className="list-group display-2">
                                {this.listItems(this.props.corrections[word])}
                                <li>
                                    <button type="button" className="btn btn-secondary btn-lg btn-block"
                                            onClick={this.props.reload}>None of the above
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
