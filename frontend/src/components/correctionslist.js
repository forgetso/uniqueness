'use strict';
import React, {Component} from 'react';

class CorrectionsList extends Component {
    constructor(props) {
        super(props);

    }

    listItems() {
        return this.props.corrections.map((correction) =>
            <li key={correction.toString()}>
                <button type="button" className="btn btn-primary btn-lg"
                        onClick={this.handleClick}>{correction}</button>
            </li>
        )
    }

    handleClick = (event) => {
        this.props.sendSpellingCorrection(this.props.response_id, event.target.innerText);
    };

    reloadComponent = (event) => {
        this.props.reloadComponent();
    };

    render() {
        return (
            <ul className="list-group">
                {this.listItems()}
                <li>
                    <button type="button" className="btn btn-primary btn-lg"
                            onClick={this.props.reload}>None of the above
                    </button>
                </li>
            </ul>)
    }
}


export default CorrectionsList;
